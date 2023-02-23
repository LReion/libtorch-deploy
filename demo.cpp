#include<filesystem>
#include<iostream>

int main(int argc, char **argv) {
    std::cout << argv[0] << std::endl;
    auto current_path = std::filesystem::current_path();
    auto target_path = current_path.append("1", "2");
    std::cout << target_path << std::endl;
}