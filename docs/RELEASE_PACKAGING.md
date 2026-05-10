# Release Packaging Checklist

Rbitnet release assets are produced by `.github/workflows/release.yml` when a `v*` tag is pushed.

## Asset Names

Current asset names are:

- `rbitnet-server-vX.Y.Z-windows-x86_64.zip`
- `rbitnet-server-vX.Y.Z-linux-x86_64.tar.gz`
- `rbitnet-server-vX.Y.Z-linux-aarch64.tar.gz`
- `rbitnet-server-vX.Y.Z-macos-x86_64.tar.gz`
- `rbitnet-server-vX.Y.Z-macos-arm64.tar.gz`

Each archive contains `rbitnet`, `rbitnet-server`, `rbitnet-runner`, and `rbitnet-proxy`.

## Checklist

1. Create and push a tag: `git tag vX.Y.Z && git push origin vX.Y.Z`.
2. Wait for the Release workflow to upload all archives.
3. Download the archives and compute SHA-256 values:

```bash
sha256sum rbitnet-server-vX.Y.Z-*.tar.gz rbitnet-server-vX.Y.Z-*.zip
```

```powershell
Get-FileHash .\rbitnet-server-vX.Y.Z-windows-x86_64.zip -Algorithm SHA256
```

4. Update `packaging/homebrew/rbitnet.rb`: version, URLs, and every `sha256`.
5. Update `packaging/winget/Rbitnet.Rbitnet.yaml`: `PackageVersion`, `InstallerUrl`, and `InstallerSha256`.
6. Smoke-test from a clean machine or container:

```bash
brew install --formula ./packaging/homebrew/rbitnet.rb
rbitnet --version
RBITNET_STUB=1 rbitnet serve
```

```powershell
winget install --manifest .\packaging\winget\Rbitnet.Rbitnet.yaml
rbitnet --version
$env:RBITNET_STUB="1"; rbitnet serve
```

7. Submit the WinGet manifest to `microsoft/winget-pkgs` and copy the Homebrew formula into the tap repository when the release assets are final.

Do not publish package metadata with placeholder SHA values or untagged URLs.
