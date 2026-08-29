{
  description = "effect-torch development environment";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
  };

  outputs = { nixpkgs, ... }:
    let
      systems = [
        "aarch64-darwin"
        "x86_64-darwin"
        "x86_64-linux"
        "aarch64-linux"
      ];
      forAllSystems = nixpkgs.lib.genAttrs systems;
      runpodctlSources = {
        aarch64-darwin = {
          suffix = "darwin-arm64";
          hash = "sha256-6l2TbA2d8j97L/ZnSAy1qDRKwpt8dpdRGCxhgPQndAg=";
        };
        x86_64-darwin = {
          suffix = "darwin-amd64";
          hash = "sha256-FrQgVRv0v6hKo3dJA5O02PRrn6uhEtD1D6JE2wSfNgY=";
        };
        x86_64-linux = {
          suffix = "linux-amd64";
          hash = "sha256-8nNVW5NZY5JeaW6V82qIPKaMXIRe/Ik9ufj3AXSchHQ=";
        };
        aarch64-linux = {
          suffix = "linux-arm64";
          hash = "sha256-djM2vdDfIqL34NqOV/UWCfwcZUZyJHOXmVQJ2wuS9RQ=";
        };
      };
      runpodctlFor = pkgs: system:
        let
          source = runpodctlSources.${system};
        in
        pkgs.stdenvNoCC.mkDerivation {
          pname = "runpodctl";
          version = "2.12.0";
          src = pkgs.fetchurl {
            url = "https://github.com/runpod/runpodctl/releases/download/v2.12.0/runpodctl-${source.suffix}";
            inherit (source) hash;
          };
          dontUnpack = true;
          installPhase = ''
            runHook preInstall
            install -Dm755 "$src" "$out/bin/runpodctl"
            runHook postInstall
          '';
          meta.mainProgram = "runpodctl";
        };
      commonPackages = pkgs: system: with pkgs; [
        nodejs_22
        corepack
        rustup
        zig
        cargo-zigbuild
        dprint
        cmake
        pkg-config
        git
        jq
        (runpodctlFor pkgs system)
        crane
      ];
    in
    {
      devShells = forAllSystems (system:
        let
          pkgs = import nixpkgs {
            inherit system;
            config.allowUnfree = true;
          };
          default = pkgs.mkShell {
            packages = commonPackages pkgs system;
          };
        in
        {
          inherit default;
        }
        // pkgs.lib.optionalAttrs (system == "x86_64-linux") (
          let
            cuda = pkgs.cudaPackages;
            cudaToolkit = pkgs.buildEnv {
              name = "effect-torch-cuda-toolkit-${cuda.cuda_nvcc.version}";
              paths = [
                cuda.cuda_nvcc
                cuda.cuda_cudart
                cuda.cccl
                cuda.cuda_nvrtc
                cuda.cuda_nvrtc.dev
                cuda.cuda_nvrtc.include
                cuda.cuda_nvrtc.lib
                cuda.libcublas
                cuda.libcublas.dev
                cuda.libcublas.include
                cuda.libcublas.lib
              ];
              pathsToLink = [
                "/bin"
                "/include"
                "/lib"
                "/nvvm"
              ];
            };
            cudaLibraryPath = pkgs.lib.makeLibraryPath [
              cuda.cuda_cudart
              cuda.cuda_nvrtc.lib
              cuda.libcublas.lib
            ];
          in
          {
            cuda = pkgs.mkShell {
              packages = commonPackages pkgs system ++ [
                cudaToolkit
                cuda.cuda_nvcc
                cuda.cuda_cudart
                cuda.cccl
                cuda.cuda_nvrtc
                cuda.libcublas
              ];

              CUDA_HOME = cudaToolkit;
              CUDA_PATH = cudaToolkit;
              CUDA_ROOT = cudaToolkit;
              CUDAToolkit_ROOT = cudaToolkit;
              EFFECT_TORCH_CUDA_ARCH = "sm_120";

              shellHook = ''
                # Expose the host NVIDIA driver without also exposing its glibc.
                driverLibraryPath="''${TMPDIR:-/tmp}/effect-torch-cuda-driver"
                hostLdconfig=$(PATH=/sbin:/usr/sbin:/usr/bin:/bin command -v ldconfig || true)
                mkdir -p "$driverLibraryPath"
                if [[ -n "$hostLdconfig" ]]; then
                  while read -r soname path; do
                    ln -sfn "$path" "$driverLibraryPath/$soname"
                  done < <("$hostLdconfig" -p | awk '/^[[:space:]]+(libcuda\.so|libnvidia-)/ { print $1, $NF }')
                fi
                export LD_LIBRARY_PATH="$driverLibraryPath:${cudaLibraryPath}''${LD_LIBRARY_PATH:+:''${LD_LIBRARY_PATH}}"
              '';
            };
          }
        ));
    };
}
