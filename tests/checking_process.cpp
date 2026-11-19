#include <iostream>
#include <chrono>
#include <thread>

template <typename Launcher>
float checking_unit(Launcher unit_fn, int warmup_iters = 3) {
    
    printf("checking step 1.\n");
    float total_ms = 0.3f;
    unit_fn();
    printf("checking step 2.\n");
    return total_ms;  // Average time per instance (ms)
}


int main() {
    
    auto unite_fn = []() {
        printf("unit instance...\n");
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    };

    float avg_time = checking_unit(unite_fn);
    printf("checking over: %.2f ms\n", avg_time);

    return 0;
}