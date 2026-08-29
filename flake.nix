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
      commonPackages = pkgs: with pkgs; [
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
        runpodctl
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
            packages = commonPackages pkgs;
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
              packages = commonPackages pkgs ++ [
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
