#pragma once

#include <algorithm>
#include <array>
#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <iostream>
#include <limits>
#include <memory>
#include <streambuf>
#include <string>
#include <system_error>
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <aclapi.h>
#ifdef _MSC_VER
#pragma comment(lib, "Advapi32.lib")
#endif
#else
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace strata::core {

// Owns one regular-file handle; stream operations never reopen its path.
// Final-component no-follow checks still require private, stable parent directories.
class RegularFileStream {
    class Buffer final : public std::streambuf {
    public:
        ~Buffer() override { std::error_code ignored; close(ignored); }
        bool open(const std::filesystem::path& path, bool writable, uint64_t limit, std::error_code& ec) {
            writable_ = writable;
            limit_ = limit;
            if (path.native().find(std::filesystem::path::value_type{}) != std::filesystem::path::string_type::npos) {
                ec = std::make_error_code(std::errc::invalid_argument);
                return false;
            }
#ifndef _WIN32
            if (limit > static_cast<uint64_t>((std::numeric_limits<off_t>::max)())) {
                ec = std::make_error_code(std::errc::value_too_large);
                return false;
            }
#endif
            if (limit > static_cast<uint64_t>((std::numeric_limits<int64_t>::max)())) {
                ec = std::make_error_code(std::errc::value_too_large);
                return false;
            }
            return native_open(path, ec);
        }
        uint64_t size() const { return size_; }
        std::error_code error() const { return error_; }
        bool close(std::error_code& ec) {
            if (is_open()) {
                if (writable_ && dirty_) sync();
                native_close();
            }
            ec = error_;
            return !ec;
        }
    protected:
        int sync() override {
            if (!is_open()) { fail(std::errc::bad_file_descriptor); return -1; }
            if (writable_ && dirty_) {
                if (!native_flush()) return -1;
                dirty_ = false;
            }
            return error_ ? -1 : 0;
        }
        int_type underflow() override {
            if (!ready(false) || position_ == size_) return traits_type::eof();
            if (read_once(&peek_, 1, position_) != 1) return traits_type::eof();
            return traits_type::to_int_type(peek_);
        }
        int_type uflow() override {
            const auto value = underflow();
            if (!traits_type::eq_int_type(value, traits_type::eof())) ++position_;
            return value;
        }
        int_type pbackfail(int_type value = traits_type::eof()) override {
            if (!ready(false) || !position_) return traits_type::eof();
            if (!traits_type::eq_int_type(value, traits_type::eof())) {
                char previous = 0;
                if (read_once(&previous, 1, position_ - 1) != 1 ||
                    !traits_type::eq_int_type(value, traits_type::to_int_type(previous))) return traits_type::eof();
            }
            --position_;
            return traits_type::not_eof(value);
        }
        std::streamsize xsgetn(char* out, std::streamsize count) override {
            if (count <= 0 || !ready(false)) return 0;
            const auto wanted = (std::min)(static_cast<uint64_t>(count), size_ - position_);
            uint64_t done = 0;
            while (done < wanted) {
                const auto chunk = static_cast<std::size_t>((std::min)(wanted - done, uint64_t{1} << 30));
                const auto read = read_once(out + done, chunk, position_);
                if (!read) break;
                done += read;
                position_ += read;
            }
            return static_cast<std::streamsize>(done);
        }
        std::streamsize xsputn(const char* data, std::streamsize count) override {
            if (count <= 0 || !ready(true)) return 0;
            if (static_cast<uint64_t>(count) > limit_ - position_) {
                fail(std::errc::no_buffer_space);
                return 0;
            }
            uint64_t done = 0;
            while (done < static_cast<uint64_t>(count)) {
                const auto chunk = static_cast<std::size_t>((std::min)(static_cast<uint64_t>(count) - done, uint64_t{1} << 30));
                const auto written = write_once(data + done, chunk, position_);
                if (!written) break;
                done += written;
                position_ += written;
                size_ = (std::max)(size_, position_);
                dirty_ = true;
            }
            return static_cast<std::streamsize>(done);
        }
        int_type overflow(int_type value = traits_type::eof()) override {
            if (traits_type::eq_int_type(value, traits_type::eof())) return traits_type::not_eof(value);
            const char byte = traits_type::to_char_type(value);
            return xsputn(&byte, 1) == 1 ? value : traits_type::eof();
        }
        pos_type seekoff(off_type offset, std::ios_base::seekdir direction, std::ios_base::openmode mode) override {
            if (((mode & std::ios_base::in) && !ready(false)) ||
                ((mode & std::ios_base::out) && !ready(true))) return pos_type(off_type(-1));
            uint64_t base = 0;
            if (direction == std::ios_base::cur) base = position_;
            else if (direction == std::ios_base::end) base = size_;
            else if (direction != std::ios_base::beg) return bad_seek();
            const uint64_t ceiling = writable_ ? limit_ : size_;
            uint64_t target = base;
            if (offset < 0) {
                const uint64_t amount = static_cast<uint64_t>(-(offset + 1)) + 1;
                if (amount > base) return bad_seek();
                target -= amount;
            } else {
                if (static_cast<uint64_t>(offset) > ceiling - base) return bad_seek();
                target += static_cast<uint64_t>(offset);
            }
            position_ = target;
            return pos_type(static_cast<off_type>(position_));
        }
        pos_type seekpos(pos_type position, std::ios_base::openmode mode) override {
            return seekoff(static_cast<off_type>(position), std::ios_base::beg, mode);
        }
    private:
        bool ready(bool writing) {
            if (error_) return false;
            if (!is_open()) { fail(std::errc::bad_file_descriptor); return false; }
            if (writing != writable_) { fail(std::errc::operation_not_permitted); return false; }
            return true;
        }
        void fail(std::errc error) { if (!error_) error_ = std::make_error_code(error); }
        void fail_native();
        pos_type bad_seek() { return pos_type(off_type(-1)); }
        bool native_open(const std::filesystem::path& path, std::error_code& ec);
        bool native_seek(uint64_t position);
        std::size_t read_once(char* out, std::size_t count, uint64_t position);
        std::size_t write_once(const char* data, std::size_t count, uint64_t position);
        bool native_flush();
        void native_close();
#ifdef _WIN32
        struct CreationSecurity {
            bool initialize(std::error_code& ec);
            bool verify(HANDLE file, std::error_code& ec);
            SECURITY_ATTRIBUTES attributes{sizeof(SECURITY_ATTRIBUTES), nullptr, FALSE};
            SECURITY_DESCRIPTOR descriptor{};
            alignas(SID) std::array<std::byte, SECURITY_MAX_SID_SIZE> sid{};
            alignas(ACL) std::array<std::byte, sizeof(ACL) + sizeof(ACCESS_ALLOWED_ACE) - sizeof(DWORD) +
                                                  SECURITY_MAX_SID_SIZE> acl{};
        };
        bool is_open() const { return handle_ != INVALID_HANDLE_VALUE; }
        HANDLE handle_ = INVALID_HANDLE_VALUE;
#else
        bool is_open() const { return handle_ >= 0; }
        int handle_ = -1;
#endif
        uint64_t limit_ = 0, size_ = 0, position_ = 0;
        bool writable_ = false, dirty_ = false;
        char peek_ = 0;
        std::error_code error_;
    };
public:
    static std::unique_ptr<RegularFileStream> create_exclusive(const std::filesystem::path& path,
                                                              uint64_t byte_limit, std::error_code& ec) {
        return open(path, byte_limit, true, ec);
    }
    static std::unique_ptr<RegularFileStream> open_read(const std::filesystem::path& path,
                                                       uint64_t byte_limit, std::error_code& ec) {
        return open(path, byte_limit, false, ec);
    }
    ~RegularFileStream() = default;
    RegularFileStream(const RegularFileStream&) = delete;
    RegularFileStream& operator=(const RegularFileStream&) = delete;
    std::istream& input() { return stream_; }
    std::ostream& output() { return stream_; }
    uint64_t size() const { return buffer_.size(); }
    std::error_code last_error() const { return buffer_.error(); }
    bool flush(std::error_code& ec) {
        const bool ok = buffer_.pubsync() == 0;
        ec = buffer_.error();
        return ok;
    }
    bool close(std::error_code& ec) { return buffer_.close(ec); }
private:
    RegularFileStream() : stream_(&buffer_) {}
    static std::unique_ptr<RegularFileStream> open(const std::filesystem::path& path, uint64_t limit,
                                                  bool writable, std::error_code& ec) {
        ec.clear();
        auto owned = std::unique_ptr<RegularFileStream>(new RegularFileStream);
        if (!owned->buffer_.open(path, writable, limit, ec)) return {};
        return owned;
    }
    Buffer buffer_;
    std::iostream stream_;
};

// Native implementation.
#ifdef _WIN32
// Microsoft: CreateFileW defaults inherit parent ACLs when no SD is supplied.
// SetSecurityDescriptorDacl(TRUE, nonnull, FALSE) and SE_DACL_PROTECTED avoid
// both NULL-DACL grants and inherited ACEs. No existing object's ACL is changed.
inline bool RegularFileStream::Buffer::CreationSecurity::initialize(std::error_code& ec) {
    HANDLE token = nullptr;
    if (!OpenThreadToken(GetCurrentThread(), TOKEN_QUERY, TRUE, &token)) {
        const auto error = GetLastError();
        if (error != ERROR_NO_TOKEN) {
            ec = std::error_code(static_cast<int>(error), std::system_category()); return false;
        }
        if (!OpenProcessToken(GetCurrentProcess(), TOKEN_QUERY, &token)) {
            ec = std::error_code(static_cast<int>(GetLastError()), std::system_category()); return false;
        }
    }
    alignas(TOKEN_USER) std::array<std::byte, sizeof(TOKEN_USER) + SECURITY_MAX_SID_SIZE> user_buffer{};
    DWORD needed = 0, error = ERROR_SUCCESS;
    if (!GetTokenInformation(token, TokenUser, user_buffer.data(), static_cast<DWORD>(user_buffer.size()), &needed))
        error = GetLastError();
    if (!CloseHandle(token) && error == ERROR_SUCCESS) error = GetLastError();
    if (error != ERROR_SUCCESS) { ec = std::error_code(static_cast<int>(error), std::system_category()); return false; }
    if (needed < sizeof(TOKEN_USER) || needed > user_buffer.size()) {
        ec = std::make_error_code(std::errc::invalid_argument); return false;
    }
    const auto user = reinterpret_cast<const TOKEN_USER*>(user_buffer.data());
    if (!IsValidSid(user->User.Sid) || GetLengthSid(user->User.Sid) > sid.size()) {
        ec = std::make_error_code(std::errc::invalid_argument); return false;
    }
    const auto caller = static_cast<PSID>(sid.data());
    const auto dacl = reinterpret_cast<PACL>(acl.data());
    if (!CopySid(static_cast<DWORD>(sid.size()), caller, user->User.Sid) ||
        !InitializeAcl(dacl, static_cast<DWORD>(acl.size()), ACL_REVISION) ||
        !AddAccessAllowedAceEx(dacl, ACL_REVISION, 0, FILE_ALL_ACCESS, caller) ||
        !InitializeSecurityDescriptor(&descriptor, SECURITY_DESCRIPTOR_REVISION) ||
        !SetSecurityDescriptorOwner(&descriptor, caller, FALSE) ||
        !SetSecurityDescriptorDacl(&descriptor, TRUE, dacl, FALSE) ||
        !SetSecurityDescriptorControl(&descriptor, SE_DACL_PROTECTED, SE_DACL_PROTECTED)) {
        ec = std::error_code(static_cast<int>(GetLastError()), std::system_category()); return false;
    }
    attributes.lpSecurityDescriptor = &descriptor;
    return true;
}
inline bool RegularFileStream::Buffer::CreationSecurity::verify(HANDLE file, std::error_code& ec) {
    PSECURITY_DESCRIPTOR actual = nullptr;
    PACL dacl = nullptr;
    PSID owner = nullptr;
    const DWORD result = GetSecurityInfo(file, SE_FILE_OBJECT, OWNER_SECURITY_INFORMATION | DACL_SECURITY_INFORMATION,
                                         &owner, nullptr, &dacl, nullptr, &actual);
    std::unique_ptr<void, decltype(&LocalFree)> release(actual, &LocalFree);
    if (result != ERROR_SUCCESS) {
        ec = std::error_code(static_cast<int>(result), std::system_category()); return false;
    }
    SECURITY_DESCRIPTOR_CONTROL control = 0;
    DWORD revision = 0;
    void* raw_ace = nullptr;
    if (!actual || !IsValidSecurityDescriptor(actual) || !owner || !IsValidSid(owner) ||
        !EqualSid(owner, sid.data()) || !dacl || !IsValidAcl(dacl) || dacl->AceCount != 1 ||
        !GetSecurityDescriptorControl(actual, &control, &revision) ||
        (control & (SE_DACL_PRESENT | SE_DACL_PROTECTED)) != (SE_DACL_PRESENT | SE_DACL_PROTECTED) ||
        !GetAce(dacl, 0, &raw_ace)) {
        ec = std::make_error_code(std::errc::permission_denied); return false;
    }
    const auto ace = static_cast<const ACCESS_ALLOWED_ACE*>(raw_ace);
    if (ace->Header.AceType != ACCESS_ALLOWED_ACE_TYPE || ace->Header.AceFlags != 0 ||
        ace->Mask != FILE_ALL_ACCESS || !IsValidSid(const_cast<DWORD*>(&ace->SidStart)) ||
        !EqualSid(const_cast<DWORD*>(&ace->SidStart), sid.data())) {
        ec = std::make_error_code(std::errc::permission_denied); return false;
    }
    return true;
}
inline void RegularFileStream::Buffer::fail_native() {
    if (!error_) error_ = std::error_code(static_cast<int>(GetLastError()), std::system_category());
}
inline bool RegularFileStream::Buffer::native_open(const std::filesystem::path& path, std::error_code& ec) {
    auto full = std::filesystem::absolute(path, ec);
    if (ec) return false;
    full.make_preferred();
    auto name = full.native();
    const bool drive = name.size() >= 3 && name[1] == L':' &&
        ((name[0] >= L'A' && name[0] <= L'Z') || (name[0] >= L'a' && name[0] <= L'z'));
    const auto colon = name.find(L':');
    if (colon != std::wstring::npos &&
        (!drive || colon != 1 || name.find(L':', colon + 1) != std::wstring::npos)) {
        ec = std::make_error_code(std::errc::invalid_argument); return false;
    }
    // Generate a disk-path namespace ourselves, rather than accepting device names.
    if (name.rfind(L"\\\\?\\", 0) == 0 || name.rfind(L"\\\\.\\", 0) == 0) {
        ec = std::make_error_code(std::errc::invalid_argument); return false;
    }
    if (name.rfind(L"\\\\", 0) == 0) name = L"\\\\?\\UNC\\" + name.substr(2);
    else if (name.size() >= 3 && name[1] == L':') name = L"\\\\?\\" + name;
    else { ec = std::make_error_code(std::errc::invalid_argument); return false; }
    const DWORD attributes = GetFileAttributesW(name.c_str());
    if (writable_) {
        if (attributes != INVALID_FILE_ATTRIBUTES) {
            ec = std::make_error_code(std::errc::file_exists); return false;
        }
        const auto error = GetLastError();
        if (error != ERROR_FILE_NOT_FOUND && error != ERROR_PATH_NOT_FOUND) {
            ec = std::error_code(static_cast<int>(error), std::system_category()); return false;
        }
    } else if (attributes == INVALID_FILE_ATTRIBUTES) {
        ec = std::error_code(static_cast<int>(GetLastError()), std::system_category()); return false;
    } else if (attributes & (FILE_ATTRIBUTE_DIRECTORY | FILE_ATTRIBUTE_REPARSE_POINT | FILE_ATTRIBUTE_DEVICE)) {
        ec = std::make_error_code(std::errc::operation_not_supported); return false;
    }
    CreationSecurity security;
    if (writable_ && !security.initialize(ec)) return false;
    const DWORD access = writable_ ? GENERIC_WRITE | FILE_READ_ATTRIBUTES | READ_CONTROL | DELETE : GENERIC_READ;
    handle_ = CreateFileW(name.c_str(), access, FILE_SHARE_READ, writable_ ? &security.attributes : nullptr,
                          writable_ ? CREATE_NEW : OPEN_EXISTING,
                          FILE_ATTRIBUTE_NORMAL | FILE_FLAG_OPEN_REPARSE_POINT, nullptr);
    if (!is_open()) { ec = std::error_code(static_cast<int>(GetLastError()), std::system_category()); return false; }
    const auto reject_created = [&]() {
        if (writable_) {
            // Delete only our CREATE_NEW object through its handle, never by path.
            // If the FS rejects disposition, report that failure; no payload was written.
            FILE_DISPOSITION_INFO remove{TRUE};
            if (!SetFileInformationByHandle(handle_, FileDispositionInfo, &remove, sizeof remove)) {
                const auto cleanup_error = GetLastError();
                std::fprintf(stderr, "regular file creation rejected (%d); disposition failed (%lu)\n",
                             ec.value(), static_cast<unsigned long>(cleanup_error));
            }
        }
        native_close();
        if (error_) std::fprintf(stderr, "regular file creation rejected (%d); close failed (%d)\n",
                                 ec.value(), error_.value());
        return false;
    };
    FILE_ATTRIBUTE_TAG_INFO info{};
    LARGE_INTEGER length{};
    if (GetFileType(handle_) != FILE_TYPE_DISK) {
        ec = std::make_error_code(std::errc::operation_not_supported); return reject_created();
    }
    if (!GetFileInformationByHandleEx(handle_, FileAttributeTagInfo, &info, sizeof info) ||
        !GetFileSizeEx(handle_, &length)) {
        ec = std::error_code(static_cast<int>(GetLastError()), std::system_category()); return reject_created();
    }
    if ((info.FileAttributes & (FILE_ATTRIBUTE_DIRECTORY | FILE_ATTRIBUTE_REPARSE_POINT | FILE_ATTRIBUTE_DEVICE)) ||
        length.QuadPart < 0 || static_cast<uint64_t>(length.QuadPart) > limit_) {
        ec = std::make_error_code(std::errc::value_too_large); return reject_created();
    }
    if (writable_ && !security.verify(handle_, ec)) return reject_created();
    size_ = static_cast<uint64_t>(length.QuadPart);
    return true;
}
inline bool RegularFileStream::Buffer::native_seek(uint64_t position) {
    LARGE_INTEGER offset{}; offset.QuadPart = static_cast<LONGLONG>(position);
    if (!SetFilePointerEx(handle_, offset, nullptr, FILE_BEGIN)) { fail_native(); return false; }
    return true;
}
#else
inline void RegularFileStream::Buffer::fail_native() {
    if (!error_) error_ = std::error_code(errno, std::generic_category());
}
inline bool RegularFileStream::Buffer::native_open(const std::filesystem::path& path, std::error_code& ec) {
    struct stat info{};
    const int status = ::lstat(path.c_str(), &info);
    if (writable_) {
        if (!status) { ec = std::make_error_code(std::errc::file_exists); return false; }
        if (errno != ENOENT) { ec = std::error_code(errno, std::generic_category()); return false; }
    } else {
        if (status) { ec = std::error_code(errno, std::generic_category()); return false; }
        if (!S_ISREG(info.st_mode)) { ec = std::make_error_code(std::errc::operation_not_supported); return false; }
    }
    const int flags = O_NOFOLLOW | O_NONBLOCK | O_CLOEXEC | (writable_ ? O_WRONLY | O_CREAT | O_EXCL : O_RDONLY);
    handle_ = ::open(path.c_str(), flags, 0600);
    if (!is_open()) { ec = std::error_code(errno, std::generic_category()); return false; }
    if (::fstat(handle_, &info)) { ec = std::error_code(errno, std::generic_category()); return false; }
    if (!S_ISREG(info.st_mode) || info.st_size < 0 || static_cast<uint64_t>(info.st_size) > limit_) {
        ec = std::make_error_code(std::errc::value_too_large); return false;
    }
    size_ = static_cast<uint64_t>(info.st_size);
    return true;
}
inline bool RegularFileStream::Buffer::native_seek(uint64_t position) {
    if (position > static_cast<uint64_t>((std::numeric_limits<off_t>::max)())) {
        fail(std::errc::value_too_large); return false;
    }
    if (::lseek(handle_, static_cast<off_t>(position), SEEK_SET) == static_cast<off_t>(-1)) {
        fail_native(); return false;
    }
    return true;
}
#endif

// Native transfers.
inline std::size_t RegularFileStream::Buffer::read_once(char* out, std::size_t count, uint64_t position) {
    if (!native_seek(position)) return 0;
#ifdef _WIN32
    DWORD done = 0;
    if (!ReadFile(handle_, out, static_cast<DWORD>(count), &done, nullptr)) { fail_native(); return 0; }
#else
    ssize_t done;
    do { done = ::read(handle_, out, count); } while (done < 0 && errno == EINTR);
    if (done < 0) { fail_native(); return 0; }
#endif
    if (!done) fail(std::errc::io_error);
    return static_cast<std::size_t>(done);
}
inline std::size_t RegularFileStream::Buffer::write_once(const char* data, std::size_t count, uint64_t position) {
    if (!native_seek(position)) return 0;
#ifdef _WIN32
    DWORD done = 0;
    if (!WriteFile(handle_, data, static_cast<DWORD>(count), &done, nullptr)) { fail_native(); return 0; }
#else
    ssize_t done;
    do { done = ::write(handle_, data, count); } while (done < 0 && errno == EINTR);
    if (done < 0) { fail_native(); return 0; }
#endif
    if (!done) fail(std::errc::io_error);
    return static_cast<std::size_t>(done);
}
inline bool RegularFileStream::Buffer::native_flush() {
#ifdef _WIN32
    if (!FlushFileBuffers(handle_)) { fail_native(); return false; }
#else
    int result;
    do { result = ::fsync(handle_); } while (result && errno == EINTR);
    if (result) { fail_native(); return false; }
#endif
    return true;
}
inline void RegularFileStream::Buffer::native_close() {
#ifdef _WIN32
    const auto handle = handle_;
    handle_ = INVALID_HANDLE_VALUE;
    if (!CloseHandle(handle)) fail_native();
#else
    const int handle = handle_;
    handle_ = -1;
    // Do not retry close(EINTR): Linux already released the descriptor and it
    // could be reused. Other POSIX implementations may retain it; close reports
    // that failure rather than claiming portable ownership recovery.
    if (::close(handle)) fail_native();
#endif
}

} // namespace strata::core
