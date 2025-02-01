#include<stdio.h>
#include<cuda.h>

//work in progress not fully optimised
 
#define B 5    //(blocksize)
#define S 10   //(sequence length)
#define d 2   //(hidden_dim)

__global__ void LA_forward(float *q, float*k, float*v,float *out,float *KV){
    __shared__ float qt[B*d];
    __shared__ float kt[B*d];
    __shared__ float vt[B*d];
    __shared__ float M[B*B]; 
    __shared__ float Temp[B*B]; 

    //loading the data in shared memory
    int idx = threadIdx.x + blockDim.x * blockIdx.x;  //columns
    int idy = threadIdx.y + blockDim.y * blockIdx.y; //rows
    int x = threadIdx.x;
    int y = threadIdx.y;
    
    if (x <= y) { // M can be made constant 
        M[x + B * y] = 1.0f;
      } else {
        M[x + B * y] = 0.0f;
      }

    if(x<d && y <B && idx<d && idy<S){
      qt[x + d*y] = q[idx + idy *d];
      kt[x + d*y] = k[idx + idy *d];
      vt[x + d*y] = v[idx + idy *d];
    }
    __syncthreads();

    float o_inter = 0.0f;
    float o_intra = 0.0f;

    if(x<d && y <B && idx<d && idy<S){
      float sum = 0;
      
      for(int i = 0;i<d;i++){
        sum += qt[i + y*d] * kt[i + y*d];
      }
      __syncthreads();
      sum = sum * M[x + B*y];
      Temp[x + y*B] = sum;
      __syncthreads();

      float value_sum =0;
      for(int j =0;j<B;j++){
        value_sum += Temp[j + y*B] * vt[x + j*B];
      }
      __syncthreads();
      o_intra += value_sum;
    }
    //compute o_inter and kv
    if(x<d && y <B && idx<d && idy<S){
      float sum_inter = 0;
      for(int i = 0;i<d;i++){
        sum_inter += qt[i + y*d] * KV[x + d*i];
      }
      __syncthreads();
      o_inter += sum_inter;

      float sum_kv = 0;
      for(int i = 0;i<d;i++){
        sum_kv += kt[i + y*d] * vt[i+y*d];
      }
      __syncthreads();
      KV[idx+idy*d] += sum_kv;
    }
    // __syncthreads();
    // if(idx<d && idy <S){
    //     printf("(%d ,%d),%.2f \n",idy,idx,o_intra);
    // }
    if(idx<d && idy<S){
      out[idx + idy*d] = o_inter + o_intra;
    }
}

int main(){
    int T = ceil((float)S/B);

    float *Q = (float*)malloc(S*d*sizeof(float)); 
    float *K = (float*)malloc(S*d*sizeof(float)); 
    float *V = (float*)malloc(S*d*sizeof(float));
    float *O = (float*)malloc(S*d*sizeof(float));

    float *q, *k, *v, *o;
    cudaMalloc((void**)&q,S*d*sizeof(float));
    cudaMalloc((void**)&k,S*d*sizeof(float));
    cudaMalloc((void**)&v,S*d*sizeof(float));
    cudaMalloc((void**)&o,S*d*sizeof(float));

    float *KV;
    cudaMalloc((void**)&KV, d*d*sizeof(float));
    cudaMemset(KV, 0, d*d*sizeof(float));

    for(int i =0; i<S;i++){
      for(int j= 0 ; j<d;j++){
        Q[j+d*i] = (float)rand() / RAND_MAX;
        K[j+d*i] = (float)rand() / RAND_MAX;
        V[j+d*i] = (float)rand() / RAND_MAX;
      }
    }
    cudaMemcpy(q,Q,S*d*sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(k,K,S*d*sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(v,V,S*d*sizeof(float),cudaMemcpyHostToDevice);

    dim3 grid (1,T);
    dim3 blocksize (d,B);
  
    LA_forward<<<grid ,blocksize>>>(q,k,v,o,KV);
    cudaMemcpy(O,o,S*d*sizeof(float),cudaMemcpyDeviceToHost);

    for(int i =0; i<2;i++){ //printing some values
        for(int j= 0 ; j<d;j++){
            printf("%.2f ",O[j + i*d]);
        }
        printf("\n");
    }
    cudaFree(q);
    cudaFree(v);
    cudaFree(k);
    cudaFree(o);
    free(Q);
    free(K);
    free(V);
    free(O);

}