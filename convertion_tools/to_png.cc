#include <iostream>
#include <fstream>
#include <vector>
#include <cstdint>
#include <png.h>
#include <filesystem> // For directory iteration
#include <regex>      // For wildcard matching

namespace fs = std::filesystem;

void save_png(const char* filename, const uint8_t* buffer, int width, int height, int bpp) {
    FILE* fp = fopen(filename, "wb");
    if (!fp) {
        std::cerr << "Failed to open file " << filename << " for writing\n";
        return;
    }

    png_structp png = png_create_write_struct(PNG_LIBPNG_VER_STRING, nullptr, nullptr, nullptr);
    if (!png) {
        std::cerr << "Failed to create PNG write struct\n";
        fclose(fp);
        return;
    }

    png_infop info = png_create_info_struct(png);
    if (!info) {
        std::cerr << "Failed to create PNG info struct\n";
        png_destroy_write_struct(&png, nullptr);
        fclose(fp);
        return;
    }

    if (setjmp(png_jmpbuf(png))) {
        std::cerr << "Failed to set PNG jump buffer\n";
        png_destroy_write_struct(&png, &info);
        fclose(fp);
        return;
    }

    png_init_io(png, fp);

    int color_type = PNG_COLOR_TYPE_RGBA;
    png_set_IHDR(png, info, width, height, 8, color_type, PNG_INTERLACE_NONE, PNG_COMPRESSION_TYPE_DEFAULT, PNG_FILTER_TYPE_DEFAULT);

    std::vector<uint8_t> rgba_buffer(width * height * 4);
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            int src_index = (y * width + x) * bpp;
            int dst_index = (y * width + x) * 4;
            rgba_buffer[dst_index + 0] = buffer[src_index + 0]; // R
            rgba_buffer[dst_index + 1] = buffer[src_index + 1]; // G
            rgba_buffer[dst_index + 2] = buffer[src_index + 2]; // B
            rgba_buffer[dst_index + 3] = 0xFF;                  // A
        }
    }

    png_bytep rows[height];
    for (int y = 0; y < height; ++y) {
        rows[y] = (png_bytep)(rgba_buffer.data() + y * width * 4);
    }

    png_set_rows(png, info, rows);
    png_write_png(png, info, PNG_TRANSFORM_IDENTITY, nullptr);

    png_destroy_write_struct(&png, &info);
    fclose(fp);
}

void convert_raw_to_png(const char* input_filename, const char* output_filename, int width, int height, int bpp) {
    std::ifstream file(input_filename, std::ios::binary);
    if (!file) {
        std::cerr << "Failed to open file " << input_filename << " for reading\n";
        return;
    }

    std::vector<uint8_t> buffer(width * height * bpp);
    file.read(reinterpret_cast<char*>(buffer.data()), buffer.size());
    if (!file) {
        std::cerr << "Failed to read data from file " << input_filename << "\n";
        return;
    }

    save_png(output_filename, buffer.data(), width, height, bpp);
}

int main(int argc, char* argv[]) {
    if (argc != 6 && argc != 5) {
        std::cerr << "Usage: " << argv[0] << " <input.raw|*.raw> <output_dir> <width> <height>\n";
        return 1;
    }

    const std::string input_pattern = argv[1];
    const std::string output_dir = argv[2];
    int width = std::stoi(argv[3]);
    int height = std::stoi(argv[4]);
    int bpp = 3;//std::stoi(argv[5]);

    if (input_pattern.find('*') != std::string::npos) {
        // Handle wildcard pattern
        std::regex pattern(std::regex_replace(input_pattern, std::regex("\\*"), ".*"));
        for (const auto& entry : fs::directory_iterator(".")) {
            if (entry.is_regular_file() && std::regex_match(entry.path().filename().string(), pattern)) {
                const std::string input_filename = entry.path().string();
                const std::string output_filename = output_dir + "/" + entry.path().stem().string() + ".png";

                std::cout << "Converting " << input_filename << " to " << output_filename << "\n";
                convert_raw_to_png(input_filename.c_str(), output_filename.c_str(), width, height, bpp);
            }
        }
    } else {
        // Handle single file
        const std::string output_filename = output_dir + "/" + fs::path(input_pattern).stem().string() + ".png";
        convert_raw_to_png(input_pattern.c_str(), output_filename.c_str(), width, height, bpp);
    }

    return 0;
}