# Security policy

## Supported versions

The latest release receives security fixes, and so does the previous minor release while it is current. Nightly builds
are not supported for security fixes; use a tagged release.

## Reporting a vulnerability

Use **[GitHub private vulnerability reporting](https://github.com/mw00/project-maya/security/advisories/new)** on this
repository. Do not open a public issue for a vulnerability.

Please include: the version (`VERSION`), the platform and GPU, the exact command or HTTP request, and what happens.
We aim to acknowledge within 72 hours and to ship a fix or mitigation within 14 days for issues in the engine, the
server or the web UI.

## Scope

In scope:

- the engine and kernels (`src/`, `include/`);
- the HTTP server and its routes (`serve/`);
- the dashboard (`serve/web/`);
- model file parsing and the quantization tools (`tools/maya_quant/`);
- the setup, launcher and installer scripts.

Out of scope: model weights and their licences, and the upstream reference code in `ref/` (report those upstream).

## Default posture

- The server binds loopback by default. Reaching it from another machine needs an explicit host setting.
- Authentication is off unless `MAYA_API_KEY` is set. Set it whenever the server leaves loopback.
- No telemetry is sent anywhere. `--report` writes a local file that the user chooses to attach to an issue.
- Responses are not stored by default. Conversation state in the dashboard is kept in the browser. The optional
  shared response history is documented in `docs/RESPONSES.md`.

## Hardening checklist for operators

- Bind to loopback or a private interface; put a reverse proxy in front for TLS if the network is not trusted.
- Set `MAYA_API_KEY` and give it only to the clients that need it.
- Run as an unprivileged user; mount model directories read-only.
- Keep the dashboard and API off the public internet unless you have reviewed the proxy, auth and CORS settings.
