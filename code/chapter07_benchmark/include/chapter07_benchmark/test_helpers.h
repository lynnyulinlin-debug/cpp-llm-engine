#pragma once

#include <string>

namespace chapter07_benchmark {

bool run_unit_checks();
bool run_e2e_checks();
std::string render_summary();
std::string render_trace();
std::string render_hotspots();
std::string render_trace_json();
std::string render_chrome_trace_json();
std::string render_multi_request_trace_json();

}  // namespace chapter07_benchmark
