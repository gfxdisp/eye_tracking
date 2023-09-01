#include <algorithm>
#include <fstream>
#include <future>
#include <iostream>
#include <string>
#include <thread>
#include <vector>
#include <filesystem>

int main(int argc, char *argv[])
{

    if (argc != 8)
    {
        std::cout << "Usage: " << argv[0]
                  << "<setup-path> <eye> <images_num> <start-number> <end-number> <num-threads> <output-path>"
                  << std::endl;
        return 1;
    }

    std::string setup_path = argv[1];
    std::string eye = argv[2];
    int num_images = std::stoi(argv[3]);
    int start_number = std::stoi(argv[4]);
    int end_number = std::stoi(argv[5]);
    int num_threads = std::stoi(argv[6]);
    std::string output_path = argv[7];

    int max_ims_per_run = 1000;

    std::vector<std::thread> threads;


    bool add_header = true;
    std::ifstream check_file(output_path + "/setups_" + eye + "/setups_params.csv");
    if (check_file.good())
    {
        add_header = false;
    }
    check_file.close();

    std::string current_folder = std::string(argv[0]);
    current_folder = current_folder.substr(0, current_folder.find_last_of("/\\"));

    std::string path_to_script = (std::filesystem::path(setup_path) / "eye_simulator" / "main.py").string();
    std::string path_to_eye = (std::filesystem::path(setup_path) / "eye_simulator" / "eye.blend").string();
    std::string path_to_resources = (std::filesystem::path(setup_path) / "resources").string();

    for (int i = start_number; i <= end_number; i++)
    {
        while (threads.size() == num_threads)
        {
            for (int j = 0; j < threads.size(); j++)
            {
                auto future = std::async(std::launch::async, &std::thread::join, &threads[j]);
                if (future.wait_for(std::chrono::seconds(1)) != std::future_status::timeout)
                {
                    threads.erase(threads.begin() + j);
                    break;
                }
            }
        }

        threads.emplace_back(
                [i, output_path, path_to_eye, path_to_script, eye, num_images, max_ims_per_run, path_to_resources]()
                {
                    for (int j = 0; j < num_images; j += max_ims_per_run)
                    {
                        int images_to_render = std::min(max_ims_per_run, num_images - j);
                        std::string command =
                                "blender " + path_to_eye + " --background --python " + path_to_script + " -- -e " +
                                eye + " -n " + std::to_string(images_to_render) + " -i " + std::to_string(i) + " -o " +
                                output_path + " -s " + path_to_resources;
                        system(command.c_str());
                    }
                });
    }

    while (threads.size() != 0)
    {
        for (int i = 0; i < threads.size(); i++)
        {
            auto future = std::async(std::launch::async, &std::thread::join, &threads[i]);
            if (future.wait_for(std::chrono::seconds(5)) != std::future_status::timeout)
            {
                threads.erase(threads.begin() + i);
                break;
            }
        }
    }

    std::ofstream output_file(output_path + "/setups_" + eye + "/setups_params.csv", std::ios::app);

    for (int i = start_number; i <= end_number; i++)
    {
        std::string i_str = std::to_string(i);
        i_str = std::string(5 - i_str.length(), '0') + i_str;
        std::ifstream input_file(output_path + "/setups_" + eye + "/" + i_str + "/setup_params.csv");

        if (i > start_number || !add_header)
        {
            std::string line;
            std::getline(input_file, line);
        }
        output_file << input_file.rdbuf();
        input_file.close();
    }
    output_file.close();

    return 0;
}