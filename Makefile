.PHONY: qari-vulkan clean

qari-vulkan:
	cmake --preset qari-vulkan-release
	cmake --build --preset qari-vulkan-release

clean:
	cmake -E rm -rf build-qari-vulkan-release
