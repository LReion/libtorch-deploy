#include "rapidjson/document.h"
#include "rapidjson/rapidjson.h"
#include "rapidjson/stringbuffer.h"
#include "rapidjson/writer.h"
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <opencv2/opencv.hpp>
#include <string>
#include <torch/script.h>
#include <torch/torch.h>

using namespace std;
using rapidjson::Document;
using rapidjson::StringBuffer;
using rapidjson::Writer;
using namespace rapidjson;

string test(const map<string, float> &m) // 注意这里的const
{
  Document document;
  Document::AllocatorType &allocator = document.GetAllocator();
  Value root(kObjectType);

  Value key(kStringType);
  Value value(kObjectType);

  for (const auto &it : m) // 注意这里要用const_iterator
  {
    key.SetString(it.first.c_str(), allocator);
    value.SetFloat(it.second);
    root.AddMember(key, value, allocator);
  }

  StringBuffer buffer;
  Writer<StringBuffer> writer(buffer);
  root.Accept(writer);
  return buffer.GetString();
}

const string classList[7] = {"adenoid",       "allergic",  "chronic",
                             "deviation",     "nasophary", "nomal",
                             "rhinosinusitis"};

int main(int argc, char *argv[]) {

  string model_path = argv[1];
  string img_path = argv[2];

  torch::NoGradGuard no_grad;

  auto model = torch::jit::load(model_path);
  model.eval();
  model.to(at::kCPU);
  cv::Mat img = cv::imread(img_path);
  cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
  cv::resize(img, img, cv::Size(448, 448), cv::INTER_LINEAR);

  torch::Tensor input_tensor =
      torch::from_blob(img.data, {1, img.rows, img.cols, 3}, torch::kByte);
  input_tensor = input_tensor.permute({0, 3, 1, 2});
  input_tensor = input_tensor.toType(torch::kFloat32);
  input_tensor = input_tensor.div(255.0);
  input_tensor[0][0].sub_(0.83984214).div_(0.18492809);
  input_tensor[0][1].sub_(0.60006026).div_(0.28001202);
  input_tensor[0][2].sub_(0.62634081).div_(0.2697814);
  auto output = model.forward({input_tensor.to(at::kCPU)}).toTensor();
  output = torch::softmax(output, 1);
  // Tensor转Vector
  vector<float> v(output.data_ptr<float>(),
                  output.data_ptr<float>() + output.numel());
  map<string, float> target_map;

  // 每类概率
  for (int i = 0; i < 7; ++i)
    target_map[classList[i]] = v[i];
  // 预测类别
  target_map["predict_class"] =
      (float)(max_element(v.begin(), v.end()) - v.begin());
  // 预测类概率
  //    target_map["predict_possibility"] = *(max_element(v.begin(), v.end()));
  fstream output_file;
  output_file.open(img_path + ".json", ios::out);
  output_file << test(target_map);
  output_file.close();
  return 0;
}