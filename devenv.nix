{
  pkgs,
  ...
}:
{
  packages = with pkgs; [
    prek
    uv
    ruff
    ty
  ];

  enterShell = ''
    if [ ! -L "$DEVENV_ROOT/.venv" ]; then
        ln -s "$DEVENV_STATE/venv/" "$DEVENV_ROOT/.venv"
    fi
    prek install --install-hooks --overwrite
    # pre-commit install --install-hooks -f
  '';

  # pre-commit.package = pkgs.prek;

  languages.python = {
    enable = true;

    uv = {
      enable = true;
      sync = {
        enable = true;
        groups = [
          "test"
        ];
      };
    };

    libraries = with pkgs; [ zlib ];
  };
}
