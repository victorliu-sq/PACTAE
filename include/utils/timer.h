#ifndef TIMER_H
#define TIMER_H

#include <sys/time.h>
#include <fstream>
#include <map>
#include <string>
#include <vector>

// Get current time in seconds (double)
inline double CurrentTime() {
  struct timeval tv;
  gettimeofday(&tv, nullptr);
  return tv.tv_sec + tv.tv_usec / 1000000.0;
}

// Record durations (name -> list of durations)
inline std::map<std::string, std::vector<double>>& TimerData() {
  static std::map<std::string, std::vector<double>> timings;
  return timings;
}

// RAII-style Scoped Timer
struct ScopedTimer {
  std::string name_;
  double start_time_;

  inline ScopedTimer(const std::string& name)
      : name_(name), start_time_(CurrentTime()) {}

  inline ~ScopedTimer() {
    double duration = (CurrentTime() - start_time_) * 1000.0; // ms
    TimerData()[name_].push_back(duration);
  }
};

// Clear recorded timings
inline void ClearTimings() {
  TimerData().clear();
}

// Write timings to a file including workload type, group size, SMP size, and GPU count
inline void WriteTimingsToFile(const std::string& filename,
                               const std::string& workload_type,
                               int num_gpus,
                               int smp_size,
                               int group_size = -1) {
  std::ofstream out(filename, std::ios::app);

  out << "workload: " << workload_type
      << ", num_gpus: " << num_gpus
      << ", smp_size: " << smp_size;

  if (group_size > 0) {
    out << ", group_size: " << group_size;
  }

  out << "\n";

  for (const auto& entry : TimerData()) {
    double total_duration = 0;
    for (double dur : entry.second) total_duration += dur;
    double avg_duration = total_duration / entry.second.size();
    out << entry.first << ": " << avg_duration << " ms (avg of "
        << entry.second.size() << " runs)\n";
  }
  out << "\n";
  out.close();

  ClearTimings();
}

// Macro for easy scoped timing
#define PROFILE_SCOPE(name) ScopedTimer scoped_timer_##__LINE__ (name)

#endif // TIMER_H
