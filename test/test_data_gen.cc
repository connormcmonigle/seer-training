#include <data_generator.h>

int main() {
  auto gen = train::data_generator("test.txt", 1000, 1024);
  gen.generate_data();
}