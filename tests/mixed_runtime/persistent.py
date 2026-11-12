"""ctypes validation adapter for the experimental persistent context ABI."""
import ctypes as c
import hashlib
from concurrent.futures import ThreadPoolExecutor
import numpy as np

P, U, Q, S, F = c.c_void_p, c.c_uint32, c.c_uint64, c.c_size_t, c.c_float
class Config(c.Structure):
    _fields_ = [('size', U), ('version', U), ('device', c.c_int32), ('gu', c.c_int32),
                ('down', c.c_int32), ('width', c.c_int32), ('intermediate', c.c_int32),
                ('slots', U), ('budget', Q)]
class Stats(c.Structure):
    _fields_ = [(n, Q) for n in ['allocated_bytes','upload_bytes','uploads','computes']]
class API(c.Structure):
    _fields_ = [('size', U), ('version', U)] + [(n,P) for n in ['create','upload','compute','stats','destroy']]

class Persistent:
    def __init__(self, lib, device, gu, down, stride):
        get = lib.maya_expert_context_api
        get.argtypes, get.restype = [U], c.POINTER(API)
        assert not get(999)
        api = get(1).contents
        assert api.version == 1 and api.size == c.sizeof(API)
        self.create = c.CFUNCTYPE(c.c_int, c.POINTER(Config), c.POINTER(P), P, S)(api.create)
        self.upload = c.CFUNCTYPE(c.c_int, P, U, Q, P, S, c.POINTER(Q), P, S)(api.upload)
        self.compute = c.CFUNCTYPE(c.c_int, P, U, P, P, P, P, F, P, P, S)(api.compute)
        self.stats = c.CFUNCTYPE(c.c_int, P, c.POINTER(Stats), P, S)(api.stats)
        self.destroy = c.CFUNCTYPE(c.c_int, c.POINTER(P), P, S)(api.destroy)
        self.cfg = Config(c.sizeof(Config),1,device,gu,down,4096,2048,8,8*stride+1024*1024)
        self.read = lib.maya_expert_context_read
        self.read.argtypes = [P,U,Q,S,P,S,P,S]
        self.read.restype = c.c_int
        self.read_checks = 0
        self.stride = stride
        self.ctx = P()
        self.err = c.create_string_buffer(512)
        bad = Config.from_buffer_copy(self.cfg); bad.budget = 1
        assert self.create(c.byref(bad),c.byref(self.ctx),self.err,len(self.err)) != 0 and not self.ctx
        self.check(self.create(c.byref(self.cfg),c.byref(self.ctx),self.err,len(self.err)))
        self.cache = [None]*8
        self.generation = [0]*8
        self.initial = self.snapshot()
        self.replacements = 0
        self.last = None

    def check(self, rc):
        if rc: raise RuntimeError(self.err.value.decode())

    def snapshot(self):
        st=Stats(); self.check(self.stats(self.ctx,c.byref(st),self.err,len(self.err)))
        return {n:int(getattr(st,n)) for n,_ in Stats._fields_}

    def run(self, slots, gens, weights, x, reject=False):
        slots=np.ascontiguousarray(slots,dtype=np.uint32); gens=np.ascontiguousarray(gens,dtype=np.uint64)
        weights=np.ascontiguousarray(weights,dtype=np.float32); x=np.ascontiguousarray(x,dtype=np.float32)
        buf=np.full(4098,-991,dtype=np.float32); err=c.create_string_buffer(512)
        rc=self.compute(self.ctx,len(slots),slots.ctypes.data,gens.ctypes.data,weights.ctypes.data,x.ctypes.data,
                        8.0,buf.ctypes.data+4,err,len(err))
        if reject:
            assert rc != 0 and np.all(buf == -991), 'rejected job changed caller output'
            return
        if rc: raise RuntimeError(err.value.decode())
        assert buf[0] == -991 and buf[-1] == -991 and np.all(np.isfinite(buf[1:-1]))
        return buf[1:-1].copy()

    def readback(self, slot, generation, offset, size, reject=False):
        buf=np.full(size+2,173,dtype=np.uint8); err=c.create_string_buffer(512)
        rc=self.read(self.ctx,slot,generation,offset,buf.ctypes.data+1,size,err,len(err))
        if reject:
            assert rc and np.all(buf==173), 'rejected read changed output'
            self.read_checks+=1
            return
        if rc: raise RuntimeError(err.value.decode())
        assert buf[0]==173 and buf[-1]==173
        self.read_checks+=1
        return buf[1:-1].copy()

    def __call__(self, blob, weights, x):
        chunks=blob.reshape(len(weights),self.stride)
        hashes=[hashlib.sha256(v).hexdigest() for v in chunks]
        slots=[]
        for chunk,h in zip(chunks,hashes):
            if h in self.cache:
                slot=self.cache.index(h)
            else:
                slot=next(i for i,old in enumerate(self.cache) if old not in hashes)
                old=self.generation[slot]; new=Q()
                # Invalid upload must leave the existing context usable.
                assert self.upload(self.ctx,8,old,chunk.ctypes.data,len(chunk),c.byref(new),self.err,len(self.err)) != 0
                assert self.upload(self.ctx,slot,old,chunk.ctypes.data,len(chunk)-1,c.byref(new),self.err,len(self.err)) != 0
                self.check(self.upload(self.ctx,slot,old,chunk.ctypes.data,len(chunk),c.byref(new),self.err,len(self.err)))
                assert new.value == old+1
                self.run([slot],[old],[1.0],x,reject=True)
                assert self.upload(self.ctx,slot,old,chunk.ctypes.data,len(chunk),c.byref(new),self.err,len(self.err)) != 0
                self.cache[slot]=h; self.generation[slot]=old+1
                self.replacements += int(old > 0)
            slots.append(slot)
        gens=[self.generation[i] for i in slots]
        before_read=self.snapshot()
        for chunk,slot,generation in zip(chunks,slots,gens):
            assert np.array_equal(self.readback(slot,generation,0,self.stride),chunk)
            assert np.array_equal(self.readback(slot,generation,131,4097),chunk[131:4228])
            self.readback(slot,generation-1,0,1,reject=True)
        self.readback(8,1,0,1,reject=True)
        self.readback(slots[0],gens[0],self.stride-1,2,reject=True)
        self.readback(slots[0],gens[0],2**64-1,1,reject=True)
        self.readback(slots[0],gens[0],0,0,reject=True)
        assert self.snapshot()==before_read, 'readback altered residency or compute counters'
        before=self.snapshot()
        y=self.run(slots,gens,weights,x)
        after=self.snapshot()
        assert after['allocated_bytes'] == self.initial['allocated_bytes']
        assert after['uploads'] == before['uploads'] and after['upload_bytes'] == before['upload_bytes']
        assert after['computes'] == before['computes']+1
        self.last=(slots,gens,weights.copy(),x.copy(),y.copy(),blob.copy())
        return y

    def lifecycle_checks(self):
        slots,gens,w,x,y,blob=self.last
        # Multiple callers share one context; library serializes their scratch usage.
        before=self.snapshot()
        with ThreadPoolExecutor(max_workers=4) as pool:
            outputs=list(pool.map(lambda _: self.run(slots,gens,w,x),range(8)))
        assert all(np.array_equal(v,y) for v in outputs)
        after=self.snapshot()
        assert after['computes'] == before['computes']+8
        assert after['uploads'] == before['uploads'] and after['allocated_bytes'] == before['allocated_bytes']
        # A second context must not borrow the first context's ownership or results.
        other=P(); self.check(self.create(c.byref(self.cfg),c.byref(other),self.err,len(self.err)))
        original=self.ctx
        try:
            self.ctx=other
            self.run([0],[gens[0]],[1.0],x,reject=True)
            zero=np.zeros(self.stride,dtype=np.uint8); generation=Q()
            self.check(self.upload(self.ctx,0,0,zero.ctypes.data,len(zero),c.byref(generation),self.err,len(self.err)))
            assert np.all(self.run([0],[generation.value],[1.0],x) == 0)
        finally:
            self.ctx=original
            self.check(self.destroy(c.byref(other),self.err,len(self.err)))
        assert np.array_equal(self.run(slots,gens,w,x),y)
        return {'stats':self.snapshot(),'replacements':self.replacements,'concurrent_jobs':8,
                'independent_context':'PASS','stale_jobs':'PASS','budget_rejection':'PASS','readback_checks':self.read_checks}

    def close(self):
        self.check(self.destroy(c.byref(self.ctx),self.err,len(self.err)))
        assert not self.ctx
        self.check(self.destroy(c.byref(self.ctx),self.err,len(self.err)))
