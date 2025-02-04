with import <nixpkgs> {};
mkShell {
	packages = [
		python312
		python312Packages.numpy
		python312Packages.graphviz
	];
}
