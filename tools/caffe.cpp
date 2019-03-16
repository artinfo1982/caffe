#ifdef WITH_PYTHON_LAYER
#include "boost/python.hpp"
namespace bp = boost::python;
#endif

#include <gflags/gflags.h>
#include <glog/logging.h>

#include <cstring>
#include <map>
#include <string>
#include <vector>

#include "boost/algorithm/string.hpp"
#include "caffe/caffe.hpp"
#include "caffe/util/signal_handler.h"

using caffe::Blob;
using caffe::Caffe;
using caffe::Layer;
using caffe::Net;
using caffe::shared_ptr;
using caffe::Solver;
using caffe::string;
using caffe::Timer;
using caffe::vector;
using std::ostringstream;

DEFINE_string(gpu, "",
              "Optional; run in GPU mode on given device IDs separated by ','."
              "Use '-gpu all' to run on all available GPUs. The effective training "
              "batch size is multiplied by the number of devices.");
DEFINE_string(solver, "",
              "The solver definition protocol buffer text file.");
DEFINE_string(model, "",
              "The model definition protocol buffer text file.");
DEFINE_string(phase, "",
              "Optional; network phase (TRAIN or TEST). Only used for 'time'.");
DEFINE_int32(level, 0,
             "Optional; network level.");
DEFINE_string(stage, "",
              "Optional; network stages (not to be confused with phase), "
              "separated by ','.");
DEFINE_string(snapshot, "",
              "Optional; the snapshot solver state to resume training.");
DEFINE_string(weights, "",
              "Optional; the pretrained weights to initialize finetuning, "
              "separated by ','. Cannot be set simultaneously with snapshot.");
DEFINE_int32(iterations, 50,
             "The number of iterations to run.");
DEFINE_string(sigint_effect, "stop",
              "Optional; action to take when a SIGINT signal is received: "
              "snapshot, stop or none.");
DEFINE_string(sighup_effect, "snapshot",
              "Optional; action to take when a SIGHUP signal is received: "
              "snapshot, stop or none.");

/* 定义全局map，k-v，key是入参的名字，value是对应的处理函数的入口 */
typedef int (*BrewFunction)();                         // BrewFunction是一个函数指针，指向一个空入参，返回值是int的函数
typedef std::map<caffe::string, BrewFunction> BrewMap; // map，key是命令参数（例train、test），value是函数指针，对应的处理函数入口
BrewMap g_brew_map;                                    // 全局变量，map类型

/*
此宏十分关键，完成注册处理函数
1. 宏里#表示字符串化，#a就变成"a"，##表示拼接，a##b就变成ab
2. 这段代码整体是一个namespace，里面一个类，以及一个这个类的全局对象
3. C++里面，全局对象的构造函数在main之前运行，全局对象的析构函数在main之后运行
4. 这个类的构造函数，是在给全局map g_brew_map赋值，将一个函数的入口地址塞入map
5. 后面每出现一次RegisterBrewFunction(xxx); 就完成一次赋值，最后在main启动前，完成所有函数的注册
*/
#define RegisterBrewFunction(func)         \
  namespace                                \
  {                                        \
  class __Registerer_##func                \
  {                                        \
  public: /* NOLINT */                     \
    __Registerer_##func()                  \
    {                                      \
      g_brew_map[#func] = &func;           \
    }                                      \
  };                                       \
  __Registerer_##func g_registerer_##func; \
  }

/*
  可以自定义函数并注册
  int XXX() {...}
  RegisterBrewFunction(XXX);
  */

static BrewFunction GetBrewFunction(const caffe::string &name)
{
  if (g_brew_map.count(name)) // c++ map的count(name)返回1/0，如果name在key列表中存在，则返回1，反之返回0
    return g_brew_map[name];  // 返回对应的处理函数入口
  LOG(FATAL) << "Unknown action: " << name << ", available caffe actions: train, test, device_query, time.";
  return nullptr;
}

// 解析gpu ids或者使用所有可用的gpu核
static void get_gpus(vector<int> *gpus)
{
  if (FLAGS_gpu == "all") // 使用所有gpu核
  {
    int count = 0;
#ifndef CPU_ONLY // GPU模式
    // cudaGetDeviceCount(int *count)，gpu数目写入count，如果函数执行成功，返回0，否则返回错误码
    CUDA_CHECK(cudaGetDeviceCount(&count));
#else // CPU模式
    NO_GPU;
#endif
    for (int i = 0; i < count; ++i)
    {
      gpus->push_back(i);
    }
  }
  else if (!FLAGS_gpu.empty()) // ids字符串非空，例如"1,3,5"
  {
    vector<string> strings;
    boost::split(strings, FLAGS_gpu, boost::is_any_of(",")); // 以逗号分割ids字符串
    for (auto s : strings)
      gpus->push_back(stoi(s));
  }
  else
    LOG(FATAL) << "Can not get any gpu ids";
}

caffe::Phase get_phase_from_flags(caffe::Phase default_value)
{
  if (FLAGS_phase == "")
    return default_value;
  if (FLAGS_phase == "TRAIN")
    return caffe::TRAIN;
  if (FLAGS_phase == "TEST")
    return caffe::TEST;
  LOG(FATAL) << "phase must be \"TRAIN\" or \"TEST\"";
  return caffe::TRAIN; // 此行无法执行到，只是为了编译不报warning
}

vector<string> get_stages_from_flags()
{
  vector<string> stages;
  boost::split(stages, FLAGS_stage, boost::is_any_of(","));
  return stages;
}

// 获取GPU信息
int device_query()
{
  LOG(INFO) << "Querying GPUs " << FLAGS_gpu;
  vector<int> gpus;
  get_gpus(&gpus);
  for (auto g : gpus)
  {
    caffe::Caffe::SetDevice(g);
    caffe::Caffe::DeviceQuery();
  }
  return 0;
}
RegisterBrewFunction(device_query);

// 信号处理
caffe::SolverAction::Enum GetRequestedAction(const std::string &flag_value)
{
  if (flag_value == "stop")
    return caffe::SolverAction::STOP;
  if (flag_value == "snapshot")
    return caffe::SolverAction::SNAPSHOT;
  if (flag_value == "none")
    return caffe::SolverAction::NONE;
  LOG(FATAL) << "Invalid signal effect \"" << flag_value << "\" was specified";
}

// 训练入口，调优模型
int train()
{
  CHECK_GT(FLAGS_solver.size(), 0) << "Need a solver definition to train."; // solver不能为空
  CHECK(!FLAGS_snapshot.size() || !FLAGS_weights.size())
      << "Give a snapshot to resume training or weights to finetune "
         "but not both."; // snapshot和weights不能同时为空
  vector<string> stages = get_stages_from_flags();
  caffe::SolverParameter solver_param;
  caffe::ReadSolverParamsFromTextFileOrDie(FLAGS_solver, &solver_param);
  // 设置level和stages
  solver_param.mutable_train_state()->set_level(FLAGS_level);
  for (auto s : stages)
    solver_param.mutable_train_state()->add_stage(s);
  // 没有在命令行指定gpu，但可以通过solver prototxt配置
  if (FLAGS_gpu.empty() && solver_param.has_solver_mode() && solver_param.solver_mode() == caffe::SolverParameter_SolverMode_GPU)
  {
    if (solver_param.has_device_id())
      FLAGS_gpu = std::to_string(solver_param.device_id());
    else
      FLAGS_gpu = "0"; // 没指定的话，使用0
  }
  vector<int> gpus;
  get_gpus(&gpus);
  if (gpus.empty())
  {
    LOG(INFO) << "Use CPU.";
    Caffe::set_mode(Caffe::CPU);
  }
  else
  {
#ifndef CPU_ONLY // GPU模式
    cudaDeviceProp device_prop;
    for (auto g : gpus)
    {
      cudaGetDeviceProperties(&device_prop, g);
      LOG(INFO) << "GPU " << g << ": " << device_prop.name;
    }
#endif
    solver_param.set_device_id(gpus[0]);
    Caffe::SetDevice(gpus[0]);
    Caffe::set_mode(Caffe::GPU);
    Caffe::set_solver_count(gpus.size());
  }
  caffe::SignalHandler signal_handler(
      GetRequestedAction(FLAGS_sigint_effect),
      GetRequestedAction(FLAGS_sighup_effect));
  if (FLAGS_snapshot.size())
    solver_param.clear_weights();
  else if (FLAGS_weights.size())
  {
    solver_param.clear_weights();
    solver_param.add_weights(FLAGS_weights);
  }
  shared_ptr<caffe::Solver<float>>
  solver(caffe::SolverRegistry<float>::CreateSolver(solver_param));
  solver->SetActionFunction(signal_handler.GetActionFunction());
  if (FLAGS_snapshot.size())
  {
    LOG(INFO) << "Resuming from " << FLAGS_snapshot;
    solver->Restore(FLAGS_snapshot.c_str());
  }
  LOG(INFO) << "Starting Optimization";
  if (gpus.size() > 1)
  {
#ifdef USE_NCCL // 多GPU场景
    caffe::NCCL<float> nccl(solver);
    nccl.Run(gpus, FLAGS_snapshot.size() > 0 ? FLAGS_snapshot.c_str() : NULL);
#else
    LOG(FATAL) << "Multi-GPU execution not available - rebuild with USE_NCCL";
#endif
  }
  else
    solver->Solve(); // 训练的主入口
  LOG(INFO) << "Train Done.";
  return 0;
}
RegisterBrewFunction(train);

// 测试入口
int test()
{
  CHECK_GT(FLAGS_model.size(), 0) << "Need a model definition to score.";
  CHECK_GT(FLAGS_weights.size(), 0) << "Need model weights to score.";
  vector<string> stages = get_stages_from_flags();

  // Set device id and mode
  vector<int> gpus;
  get_gpus(&gpus);
  if (gpus.size() != 0)
  {
    LOG(INFO) << "Use GPU with device ID " << gpus[0];
#ifndef CPU_ONLY
    cudaDeviceProp device_prop;
    cudaGetDeviceProperties(&device_prop, gpus[0]);
    LOG(INFO) << "GPU device name: " << device_prop.name;
#endif
    Caffe::SetDevice(gpus[0]);
    Caffe::set_mode(Caffe::GPU);
  }
  else
  {
    LOG(INFO) << "Use CPU.";
    Caffe::set_mode(Caffe::CPU);
  }
  // Instantiate the caffe net.
  Net<float> caffe_net(FLAGS_model, caffe::TEST, FLAGS_level, &stages);
  caffe_net.CopyTrainedLayersFrom(FLAGS_weights);
  LOG(INFO) << "Running for " << FLAGS_iterations << " iterations.";

  vector<int> test_score_output_id;
  vector<float> test_score;
  float loss = 0;
  for (int i = 0; i < FLAGS_iterations; ++i)
  {
    float iter_loss;
    const vector<Blob<float> *> &result =
        caffe_net.Forward(&iter_loss);
    loss += iter_loss;
    int idx = 0;
    for (int j = 0; j < result.size(); ++j)
    {
      const float *result_vec = result[j]->cpu_data();
      for (int k = 0; k < result[j]->count(); ++k, ++idx)
      {
        const float score = result_vec[k];
        if (i == 0)
        {
          test_score.push_back(score);
          test_score_output_id.push_back(j);
        }
        else
        {
          test_score[idx] += score;
        }
        const std::string &output_name = caffe_net.blob_names()[caffe_net.output_blob_indices()[j]];
        LOG(INFO) << "Batch " << i << ", " << output_name << " = " << score;
      }
    }
  }
  loss /= FLAGS_iterations;
  LOG(INFO) << "Loss: " << loss;
  for (int i = 0; i < test_score.size(); ++i)
  {
    const std::string &output_name = caffe_net.blob_names()[caffe_net.output_blob_indices()[test_score_output_id[i]]];
    const float loss_weight = caffe_net.blob_loss_weights()[caffe_net.output_blob_indices()[test_score_output_id[i]]];
    std::ostringstream loss_msg_stream;
    const float mean_score = test_score[i] / FLAGS_iterations;
    if (loss_weight)
    {
      loss_msg_stream << " (* " << loss_weight
                      << " = " << loss_weight * mean_score << " loss)";
    }
    LOG(INFO) << output_name << " = " << mean_score << loss_msg_stream.str();
  }

  return 0;
}
RegisterBrewFunction(test);

// 压测一个模型的执行时长
int time()
{
  CHECK_GT(FLAGS_model.size(), 0) << "Need a model definition to time.";
  caffe::Phase phase = get_phase_from_flags(caffe::TRAIN);
  vector<string> stages = get_stages_from_flags();

  // Set device id and mode
  vector<int> gpus;
  get_gpus(&gpus);
  if (gpus.size() != 0)
  {
    LOG(INFO) << "Use GPU with device ID " << gpus[0];
    Caffe::SetDevice(gpus[0]);
    Caffe::set_mode(Caffe::GPU);
  }
  else
  {
    LOG(INFO) << "Use CPU.";
    Caffe::set_mode(Caffe::CPU);
  }
  // Instantiate the caffe net.
  Net<float> caffe_net(FLAGS_model, phase, FLAGS_level, &stages);

  // Do a clean forward and backward pass, so that memory allocation are done
  // and future iterations will be more stable.
  LOG(INFO) << "Performing Forward";
  // Note that for the speed benchmark, we will assume that the network does
  // not take any input blobs.
  float initial_loss;
  caffe_net.Forward(&initial_loss);
  LOG(INFO) << "Initial loss: " << initial_loss;
  LOG(INFO) << "Performing Backward";
  caffe_net.Backward();

  const vector<shared_ptr<Layer<float>>> &layers = caffe_net.layers();
  const vector<vector<Blob<float> *>> &bottom_vecs = caffe_net.bottom_vecs();
  const vector<vector<Blob<float> *>> &top_vecs = caffe_net.top_vecs();
  const vector<vector<bool>> &bottom_need_backward =
      caffe_net.bottom_need_backward();
  LOG(INFO) << "*** Benchmark begins ***";
  LOG(INFO) << "Testing for " << FLAGS_iterations << " iterations.";
  Timer total_timer;
  total_timer.Start();
  Timer forward_timer;
  Timer backward_timer;
  Timer timer;
  std::vector<double> forward_time_per_layer(layers.size(), 0.0);
  std::vector<double> backward_time_per_layer(layers.size(), 0.0);
  double forward_time = 0.0;
  double backward_time = 0.0;
  for (int j = 0; j < FLAGS_iterations; ++j)
  {
    Timer iter_timer;
    iter_timer.Start();
    forward_timer.Start();
    for (int i = 0; i < layers.size(); ++i)
    {
      timer.Start();
      layers[i]->Forward(bottom_vecs[i], top_vecs[i]);
      forward_time_per_layer[i] += timer.MicroSeconds();
    }
    forward_time += forward_timer.MicroSeconds();
    backward_timer.Start();
    for (int i = layers.size() - 1; i >= 0; --i)
    {
      timer.Start();
      layers[i]->Backward(top_vecs[i], bottom_need_backward[i],
                          bottom_vecs[i]);
      backward_time_per_layer[i] += timer.MicroSeconds();
    }
    backward_time += backward_timer.MicroSeconds();
    LOG(INFO) << "Iteration: " << j + 1 << " forward-backward time: "
              << iter_timer.MilliSeconds() << " ms.";
  }
  LOG(INFO) << "Average time per layer: ";
  for (int i = 0; i < layers.size(); ++i)
  {
    const caffe::string &layername = layers[i]->layer_param().name();
    LOG(INFO) << std::setfill(' ') << std::setw(10) << layername << "\tforward: " << forward_time_per_layer[i] / 1000 / FLAGS_iterations << " ms.";
    LOG(INFO) << std::setfill(' ') << std::setw(10) << layername << "\tbackward: " << backward_time_per_layer[i] / 1000 / FLAGS_iterations << " ms.";
  }
  total_timer.Stop();
  LOG(INFO) << "Average Forward pass: " << forward_time / 1000 / FLAGS_iterations << " ms.";
  LOG(INFO) << "Average Backward pass: " << backward_time / 1000 / FLAGS_iterations << " ms.";
  LOG(INFO) << "Average Forward-Backward: " << total_timer.MilliSeconds() / FLAGS_iterations << " ms.";
  LOG(INFO) << "Total Time: " << total_timer.MilliSeconds() << " ms.";
  LOG(INFO) << "*** Benchmark ends ***";
  return 0;
}
RegisterBrewFunction(time);

int main(int argc, char **argv)
{
  FLAGS_alsologtostderr = 1;                          // 日志重定向到文件的同时，也打印到控制台
  FLAGS_colorlogtostderr = 1;                         // 根据日志级别，颜色区分显示
  FLAGS_stderrthreshold = 2;                          // 控制台显示日志的阈值，0-3，对应INFO-FATAL，2对应ERROR
  FLAGS_stop_logging_if_full_disk = 1;                // 磁盘满就不再写日志了
  gflags::SetVersionString(AS_STRING(CAFFE_VERSION)); // 从cmake中获取版本信息
  gflags::SetUsageMessage("command line brew\n"
                          "usage: caffe <command> <args>\n\n"
                          "commands:\n"
                          "  train           train or finetune a model\n"
                          "  test            score a model\n"
                          "  device_query    show GPU diagnostic information\n"
                          "  time            benchmark model execution time");
  caffe::GlobalInit(&argc, &argv);
  if (argc == 2) // 只到command（例如train）这一级，后面的args（例如-solver），由前面定义的gflags项负责解析
  {
#ifdef WITH_PYTHON_LAYER
    try
    {
#endif
      // 将命令行第二个参数command字符串送入GetBrewFunction，返回BrewFunction类型
      // BrewFunction是一个函数指针，指向一个空入参，返回值是int的函数，所以GetBrewFunction得到的是一个函数的入口
      // 入参：train、test、device_query、time
      // 对应的处理函数：int train()、int test()、int device_query()、int time()
      // 作者实现了一个string到函数入口的映射，十分巧妙
      return GetBrewFunction(caffe::string(argv[1]))();
#ifdef WITH_PYTHON_LAYER
    }
    catch (bp::error_already_set)
    {
      PyErr_Print();
      return 1;
    }
#endif
  }
  else
    gflags::ShowUsageWithFlagsRestrict(argv[0], "tools/caffe"); // 打印usage信息
}
