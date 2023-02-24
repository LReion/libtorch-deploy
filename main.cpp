// Add comments for the following code:
#include "rapidjson/document.h"     // Include the rapidjson document header
#include "rapidjson/rapidjson.h"    // Include the rapidjson global header
#include "rapidjson/stringbuffer.h" // Include the StringBuffer header
#include "rapidjson/writer.h"       // Include the Writer header
#include <filesystem>               // Include the filesystem library
#include <fstream>                  // Include the file stream library
#include <iostream>                 // Include the iostream library
#include <map>                      // Include the map library
#include <opencv2/opencv.hpp>       // Include OpenCV
#include <string>                   // Include the string library
#include <torch/script.h>           // Include PyTorch C++ API
#include <torch/torch.h>            // Include Torch library

using namespace std;           // Use standard namespace
using rapidjson::Document;     // Use Document class from rapid json
using rapidjson::StringBuffer; // Use StringBuffer class from rapidjson
using rapidjson::Writer;       // Use Writer class from rapid json
using namespace rapidjson;     // Use rapidjson namespace

// Function to convert a map of strings and floats to a json string
string test(const map<string, float> &m) // Note const keyword here
{
    Document document; // Create a new rapidjson Document object
    Document::AllocatorType &allocator =
            document.GetAllocator(); // Get a reference to the allocator
    Value root(kObjectType);     // Create a new value object of type 'object'

    Value key(kStringType);   // Create a new value object of type 'string'
    Value value(kObjectType); // Create a new value object of type 'object'

    // Loop through the map and populate the json object
    for (const auto &it: m) // Note const_iterator
    {
        key.SetString(it.first.c_str(), allocator); // Set Key
        value.SetFloat(it.second);                  // Set Float Value
        root.AddMember(key, value, allocator);      // Add member to the root object
    }

    StringBuffer buffer;                 // Create a new StringBuffer
    Writer<StringBuffer> writer(buffer); // Create a new Writer
    root.Accept(writer);                 // Accept the Writer
    return buffer.GetString();           // Return the json string
}

// Array of class names used for mapping
const string classList[7] = {"adenoid", "allergic", "chronic",
                             "deviation", "nasophary", "nomal",
                             "rhinosinusitis"};

int main(int argc, char *argv[]) {
    // Main function that passes command line arguments
    // and uses them to run the program

    string model_path = argv[1]; // First argument is the model path
    string img_path = argv[2];   // Second argument is the image path

    torch::NoGradGuard no_grad; // Create a no gradient guard

    auto model =
            torch::jit::load(model_path);   // Load the model from the given path
    model.eval();                       // Set the model to eval mode
    model.to(at::kCPU);                 // Force model to run on CPU
    cv::Mat img = cv::imread(img_path); // Read the image using the cv library
    cv::cvtColor(img, img, cv::COLOR_BGR2RGB); // Convert the colorspace to RGB
    cv::resize(img, img, cv::Size(448, 448),
               cv::INTER_LINEAR); // Resize the image to 448 x 448

    // Convert image data to a torch tensor
    torch::Tensor input_tensor =
            torch::from_blob(img.data, {1, img.rows, img.cols, 3}, torch::kByte);
    input_tensor = input_tensor.permute(
            {0, 3, 1, 2}); // Permute the dimensions of the tensor
    input_tensor = input_tensor.toType(torch::kFloat32); // Convert to float type
    input_tensor = input_tensor.div(255.0);              // Normalize the values

    // Subtract the mean values from each channel in the image
    input_tensor[0][0].sub_(0.83984214).div_(0.18492809);
    input_tensor[0][1].sub_(0.60006026).div_(0.28001202);
    input_tensor[0][2].sub_(0.62634081).div_(0.2697814);

    // Forward pass the model to get the output
    auto output = model.forward({input_tensor.to(at::kCPU)}).toTensor();

    // Convert output Tensor to a vector
    vector v(output.data_ptr<float>(), output.data_ptr<float>() + output.numel());

    // Create map object, that is later passed to the test method
    map<string, float> target_map;

    // Set all individual class probabilities in the map
    for (int i = 0; i < 7; ++i)
        target_map[classList[i]] = v[i];

    // Predict class and set into the map
    target_map["predict_class"] =
            (float) (max_element(v.begin(), v.end()) - v.begin());

    // Open and write json file with results
    fstream output_file;
    output_file.open(img_path + ".json", ios::out);
    output_file << test(target_map);
    output_file.close();
    return 0;
}
