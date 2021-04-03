#include <stdio.h>

void saxpy(int n, float a, float* x, float* y) {
  for (int i = 0; i < n; i++) {
    y[i] = a*x[i] + y[i];
  }
} 

int main(int argc, char** argv) {
  float x[4] = {1, 2, 3, 4};
  float y[4] = {0, 0, 0, 0};
  saxpy(4, 2, x, y);
  for (int i = 0; i < 4; i++) {
    printf("%f ", y[i]);
  }
  printf("\n");
  return 0;
}
