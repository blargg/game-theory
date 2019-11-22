{ log ? "warn"}:
with import <nixpkgs> {};
let
  # extra packages for development
  devPackages = [
    rls           # Language Server
    rustfmt
    rustPackages.clippy
    cargo-audit
  ];
  crateName="game-theory";
in
mkShell {
  buildInputs = [
    cargo
    rustc
  ]
  ++ devPackages;

  RUST_BACKTRACE = 1;
  RUST_LOG = "${crateName}=${log}";
}
