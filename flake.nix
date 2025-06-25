# Copyright (C) 2025 Paul Chambaz
# This file is part of isir-internship
{
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };
  outputs = {
    self,
    nixpkgs,
    flake-utils,
  }:
    flake-utils.lib.eachDefaultSystem (
      system: let
        pkgs = import nixpkgs {
          inherit system;
          config = {allowUnfree = true;};
        };
      in {
        devShells.default = pkgs.mkShell {
          packages = with pkgs; [
            just
            typst
            uv
            ruff
            ty
            python313Packages.tkinter
            texlive.combined.scheme-full
            mujoco
            imagemagick
          ];

          env = with pkgs; {
            LD_LIBRARY_PATH = lib.makeLibraryPath [
              stdenv.cc.cc.lib
              libz
              tk
              tcl
              xorg.libX11
              xorg.libXext
              xorg.libXrender
              xorg.libXrandr
              xorg.libXi
              xorg.libXcursor
              wayland
              libxkbcommon
              libGL
              libglvnd
              fontconfig
              freetype
              blas
              lapack
              libffi
              openssl
              sqlite
              zlib
              gfortran.cc.lib
              swig
            ];
            MPLBACKEND = "TkAgg";
          };

          shellHook = ''
            uv sync --quiet --dev
            source .venv/bin/activate
          '';
        };
      }
    );
}
