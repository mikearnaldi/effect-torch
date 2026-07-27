{
  description = "effect-torch development environment";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
  };

  outputs = { nixpkgs, ... }:
    let
      forAllSystems = nixpkgs.lib.genAttrs [
        "aarch64-darwin"
        "x86_64-darwin"
        "x86_64-linux"
        "aarch64-linux"
      ];
    in
    {
      devShells = forAllSystems (system:
        let
          pkgs = import nixpkgs { inherit system; };
        in
        {
          default = pkgs.mkShell {
            packages = with pkgs; [
              nodejs_22
              corepack
              rustc
              cargo
              rustfmt
              clippy
              rust-analyzer
              cmake
              pkg-config
            ] ++ pkgs.lib.optionals pkgs.stdenv.isDarwin [
              libiconv
            ];
          };
        });
    };
}
