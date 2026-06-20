{
  description = "Development shell for AI Python learning labs";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
  };

  outputs = { nixpkgs, ... }:
    let
      systems = [
        "x86_64-linux"
        "aarch64-linux"
        "aarch64-darwin"
        "x86_64-darwin"
      ];
      forAllSystems = nixpkgs.lib.genAttrs systems;
    in {
      devShells = forAllSystems (system:
        let
          pkgs = import nixpkgs { inherit system; };
        in {
          default = pkgs.mkShell {
            packages = [
              pkgs.python3
              pkgs.uv
            ];

            shellHook = ''
              echo "learning-ai-python dev shell"
              echo "Try: python3 projects/local-agent-eval/test_app.py"
              echo "Try: python3 projects/llm-tool-calling-lab/test_agent.py"
              echo "Try: python3 projects/mcp-stdio-tool-server/test_server.py"
            '';
          };
        });
    };
}
