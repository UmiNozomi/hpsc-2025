#include <cstdlib>
#include <cstdio>
#include <fstream>
#include <vector>
#include <iostream>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#define IDX(i,j) ((j)*nx + (i))

typedef std::vector<std::vector<float>> matrix;

__global__ void compute_b(float *b, const float *u, const float *v, float dx, float dy, int nx, int ny, float rho, float dt) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if (i < nx && j < ny) {
        
        float du_dx = (u[IDX(i+1,j)] - u[IDX(i-1,j)]) / (2 * dx);
        float dv_dy = (v[IDX(i,j+1)] - v[IDX(i,j-1)]) / (2 * dy);
        
        float div = du_dx + dv_dy;
        
        float nonlinear = du_dx * div;
        
        b[IDX(i,j)] = rho * (div / dt - nonlinear);
    }
}

__global__ void compute_pressure(float *p, const float *pn, const float *b, float dx, float dy, int nx, int ny) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if (i > 0 && i < nx-1 && j > 0 && j < ny-1) {
        p[IDX(i,j)] = ((pn[IDX(i+1,j)] + pn[IDX(i-1,j)]) * dy * dy + 
                       (pn[IDX(i,j+1)] + pn[IDX(i,j-1)]) * dx * dx - 
                       b[IDX(i,j)] * dx * dx * dy * dy) / 
                      (2 * (dx * dx + dy * dy));
    }
}

__global__ void compute_velocity(float *u, float *v, const float *un, const float *vn, 
                               const float *p, float dx, float dy, float dt, float nu, float rho,
                               int nx, int ny) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if (i > 0 && i < nx-1 && j > 0 && j < ny-1) {
        u[IDX(i,j)] = un[IDX(i,j)] - 
                      un[IDX(i,j)] * dt / dx * (un[IDX(i,j)] - un[IDX(i-1,j)]) -
                      vn[IDX(i,j)] * dt / dy * (un[IDX(i,j)] - un[IDX(i,j-1)]) -
                      dt / (2 * rho * dx) * (p[IDX(i+1,j)] - p[IDX(i-1,j)]) +
                      nu * (dt / (dx * dx) * (un[IDX(i+1,j)] - 2 * un[IDX(i,j)] + un[IDX(i-1,j)]) +
                            dt / (dy * dy) * (un[IDX(i,j+1)] - 2 * un[IDX(i,j)] + un[IDX(i,j-1)]));
                            
        v[IDX(i,j)] = vn[IDX(i,j)] -
                      un[IDX(i,j)] * dt / dx * (vn[IDX(i,j)] - vn[IDX(i-1,j)]) -
                      vn[IDX(i,j)] * dt / dy * (vn[IDX(i,j)] - vn[IDX(i,j-1)]) -
                      dt / (2 * rho * dy) * (p[IDX(i,j+1)] - p[IDX(i,j-1)]) +
                      nu * (dt / (dx * dx) * (vn[IDX(i+1,j)] - 2 * vn[IDX(i,j)] + vn[IDX(i-1,j)]) +
                            dt / (dy * dy) * (vn[IDX(i,j+1)] - 2 * vn[IDX(i,j)] + vn[IDX(i,j-1)]));
    }
}

__global__ void update_boundaries(float *u, float *v, int nx, int ny) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    
    if (idx < nx) {
        // Bottom wall
        u[IDX(idx,0)] = 0;
        v[IDX(idx,0)] = 0;
        
        // Top wall
        u[IDX(idx,ny-1)] = 1;  // Moving lid
        v[IDX(idx,ny-1)] = 0;
    }
    
    if (idx < ny) {
        // Left wall
        u[IDX(0,idx)] = 0;
        v[IDX(0,idx)] = 0;
        
        // Right wall
        u[IDX(nx-1,idx)] = 0;
        v[IDX(nx-1,idx)] = 0;
    }
}

int main() {
    int nx = 41;
    int ny = 41;
    int nt = 500;
    int nit = 50;
    float dx = 2.f / (nx - 1);
    float dy = 2.f / (ny - 1);
    float dt = .01f;
    float rho = 1.f;
    float nu = .02f;

    std::vector<float> h_u(nx * ny), h_v(nx * ny), h_p(nx * ny);
    std::vector<float> h_un(nx * ny), h_vn(nx * ny), h_pn(nx * ny), h_b(nx * ny);

    std::fill(h_u.begin(), h_u.end(), 0.f);
    std::fill(h_v.begin(), h_v.end(), 0.f);
    std::fill(h_p.begin(), h_p.end(), 0.f);
    std::fill(h_b.begin(), h_b.end(), 0.f);

    float *d_u, *d_v, *d_p, *d_b, *d_un, *d_vn, *d_pn;
    cudaMalloc(&d_u, nx * ny * sizeof(float));
    cudaMalloc(&d_v, nx * ny * sizeof(float));
    cudaMalloc(&d_p, nx * ny * sizeof(float));
    cudaMalloc(&d_b, nx * ny * sizeof(float));
    cudaMalloc(&d_un, nx * ny * sizeof(float));
    cudaMalloc(&d_vn, nx * ny * sizeof(float));
    cudaMalloc(&d_pn, nx * ny * sizeof(float));

    cudaMemcpy(d_u, h_u.data(), nx * ny * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_v, h_v.data(), nx * ny * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_p, h_p.data(), nx * ny * sizeof(float), cudaMemcpyHostToDevice);

    dim3 block(16, 16);
    dim3 grid((nx + block.x - 1) / block.x, (ny + block.y - 1) / block.y);
    dim3 block_1d(256);
    dim3 grid_1d((std::max(nx, ny) + block_1d.x - 1) / block_1d.x);

    std::ofstream ufile("u.dat"), vfile("v.dat"), pfile("p.dat");

    for (int n = 0; n < nt; n++) {
        compute_b<<<grid, block>>>(d_b, d_u, d_v, dx, dy, nx, ny, rho, dt);
        
        for (int it = 0; it < nit; it++) {
            cudaMemcpy(d_pn, d_p, nx * ny * sizeof(float), cudaMemcpyDeviceToDevice);
            compute_pressure<<<grid, block>>>(d_p, d_pn, d_b, dx, dy, nx, ny);
        }

        cudaMemcpy(d_un, d_u, nx * ny * sizeof(float), cudaMemcpyDeviceToDevice);
        cudaMemcpy(d_vn, d_v, nx * ny * sizeof(float), cudaMemcpyDeviceToDevice);

        compute_velocity<<<grid, block>>>(d_u, d_v, d_un, d_vn, d_p, dx, dy, dt, nu, rho, nx, ny);
        
        update_boundaries<<<grid_1d, block_1d>>>(d_u, d_v, nx, ny);

        if (n % 10 == 0) {
            cudaMemcpy(h_u.data(), d_u, nx * ny * sizeof(float), cudaMemcpyDeviceToHost);
            cudaMemcpy(h_v.data(), d_v, nx * ny * sizeof(float), cudaMemcpyDeviceToHost);
            cudaMemcpy(h_p.data(), d_p, nx * ny * sizeof(float), cudaMemcpyDeviceToHost);

            for (int j = 0; j < ny; j++) {
                for (int i = 0; i < nx; i++) {
                    ufile << h_u[IDX(i,j)] << " ";
                    vfile << h_v[IDX(i,j)] << " ";
                    pfile << h_p[IDX(i,j)] << " ";
                }
            }
            ufile << "\n";
            vfile << "\n";
            pfile << "\n";
        }
    }

    cudaFree(d_u);
    cudaFree(d_v);
    cudaFree(d_p);
    cudaFree(d_b);
    cudaFree(d_un);
    cudaFree(d_vn);
    cudaFree(d_pn);

    ufile.close();
    vfile.close();
    pfile.close();

    return 0;
}
