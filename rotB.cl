
#define NL %(block_size)d
#define NX %(nx)d
#define NY %(ny)d
#define NZ %(nz)d
    
#define Xs(i,j,k) X[k+NZ*j+NY*NZ*i]

float3 Cross(float3 U, float3 V)
{
    float3 Y;
    Y.x = U.y*V.z - U.z*V.y;
    Y.y = U.z*V.x - U.x*V.z;
    Y.z = U.x*V.y - U.y*V.x;
    
    return Y;
}



void Deriv(__global float3* X, uchar d, uchar m, __global float3* Y)
{
    unsigned int ix = get_global_id(0);
    unsigned int iy = get_global_id(1);
    unsigned int iz = get_global_id(2);
    
    unsigned int nx = get_global_size(0);
    unsigned int ny = get_global_size(1);
    unsigned int nz = get_global_size(2);
   
    unsigned int idx = iz+nz*iy+ny*nz*ix;
 
    unsigned int lx = get_local_id(0)+1;
    unsigned int ly = get_local_id(1)+1;
    unsigned int lz = get_local_id(2)+1;
    
    unsigned int nl = NL+2;
    
    unsigned int ldx = lz+nl*ly+nl*nl*lx;

    __local float3 Xl[(NL+2)*(NL+2)*(NL+2)];   
    Xl[ldx] = X[idx];
    if ((ix != 0) && (ix != NX-1)) {
        if (lx == 1) Xl[ldx-nl*nl] = X[idx-ny*nz];
        if (lx == nl-2) Xl[ldx+nl*nl] = X[idx+ny*nz];
    }
    if ((iy != 0) && (iy != NY-1)) {
        if (ly == 1) Xl[ldx-nl] = X[idx-nz];
        if (ly == nl-2) Xl[ldx+nl] = X[idx+nz];
    }
    if ((iz != 0) && (iz != NZ-1)) {
        if (lz == 1) Xl[ldx-1] = X[idx-1];
        if (lz == nl-2) Xl[ldx+1] = X[idx+1];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    
    unsigned int ii;
    unsigned int ng;
    unsigned int ll;
    unsigned int bg;
    unsigned int bl;
    
    if (d == 0) {
        ng = nx;
        ii = ix;
        bg = ny*nz;
        ll = lx;
        bl = nl*nl;
    }
    
    if (d == 1) {
        ng = ny;
        ii = iy;
        bg = nz;
        ll = ly;
        bl = nl;
    }
    
    if (d == 2) {
        ng = nz;
        ii = iz;
        bg = 1;
        ll = lz;
        bl = 1;
    }
    
    
        
    if ((ii != 0) && (ii != (ng-1))) { 
        if (m == 1) Y[idx] = Xl[ldx+bl]*(float3)(0.5) - Xl[ldx-bl]*(float3)(0.5);
        if (m == 2) Y[idx] = Xl[ldx+bl] + Xl[ldx-bl] - (float3)(2)*Xl[ldx];              
    } else if (ii == 0) {
        if (m == 1) Y[idx] = -(float3)(1.5)*Xl[ldx] + (float3)(2)*Xl[ldx+bl] - (float3)(0.5)*Xl[ldx+2*bl]; 
        if (m == 2) Y[idx] = (float3)(2)*Xl[ldx] - (float3)(5)*Xl[ldx+bl] + (float3)(4)*Xl[ldx+2*bl] - Xl[ldx+3*bl];
    } else if (ii == (ng-1)) {
        if (m == 1) Y[idx] = (float3)(1.5)*Xl[ldx] - (float3)(2)*Xl[ldx-bl] + (float3)(0.5)*Xl[ldx-2*bl];
        if (m == 2) Y[idx] = (float3)(2)*Xl[ldx] - (float3)(5)*Xl[ldx-bl] + (float3)(4)*Xl[ldx-2*bl] - Xl[ldx-3*bl];           
    }

    barrier(CLK_LOCAL_MEM_FENCE);
}




__kernel
void Copy(__global float3* X, __global float3* Y)
{
    unsigned int ix = get_global_id(0);
    unsigned int iy = get_global_id(1);
    unsigned int iz = get_global_id(2);
    
    unsigned int nx = get_global_size(0);
    unsigned int ny = get_global_size(1);
    unsigned int nz = get_global_size(2);

    unsigned int idx = iz+nz*iy+ny*nz*ix;
    
    Y[idx] = X[idx];    
}


__kernel __attribute__((reqd_work_group_size(NL,NL,NL)))
void Hessian(__global float3* X)
{    
  
    unsigned int ix = get_global_id(0);
    unsigned int iy = get_global_id(1);
    unsigned int iz = get_global_id(2);

    //unsigned int idx = iz+nz*iy+ny*nz*ix;
 
    unsigned int lx = get_local_id(0);
    unsigned int ly = get_local_id(1);
    unsigned int lz = get_local_id(2);
    
    unsigned int nl = NL+2;
    
    //unsigned int ldx = lz+nl*ly+nl*nl*lx;
    
    __local float3 Xl[NL+2][NL+2][NL+2];
    
    Xl[lx+1][ly+1][lz+1] = Xs(ix,iy,iz);
    
    if ((ix != 0) && (ix != NX-1)) {
        if (lx == 0) Xl[0][ly+1][lz+1] = Xs(ix-1,iy,iz);
        if (lx == NL-1) Xl[NL+1][ly+1][lz+1] = Xs(ix+1,iy,iz);
    }
    if ((iy != 0) && (iy != NY-1)) {
        if (ly == 0) Xl[lx+1][0][lz+1] = Xs(ix,iy-1,iz);
        if (ly == NL-1) Xl[lx+1][NL+1][lz+1] = Xs(ix,iy+1,iz);
    }
    if ((iz != 0) && (iz != NZ-1)) {
        if (lz == 0) Xl[0][ly+1][lz+1] = Xs(ix-1,iy,iz);
        if (lz == NL-1) Xl[NL+1][ly+1][lz+1] = Xs(ix+1,iy,iz);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    
   
}






//--------------------------------------------------

__kernel __attribute__((reqd_work_group_size(NL,NL,NL)))
void Jacobian(__global float3* X, __global float3* J)
{    
    unsigned int nx = get_global_size(0);
    unsigned int ny = get_global_size(1);
    unsigned int nz = get_global_size(2);
 
    unsigned int i;
    for (i = 0; i < 3; i++) Deriv(X, i, 1, &J[i*nx*ny*nz]);    
}







__kernel __attribute__((reqd_work_group_size(NL,NL,NL)))
void Force(__global float3* X, __global float3* J, __global float3* B, __global float3* Ix, __global float3* F)
{
    unsigned int ix = get_global_id(0);
    unsigned int iy = get_global_id(1);
    unsigned int iz = get_global_id(2);
    
    unsigned int nx = get_global_size(0);
    unsigned int ny = get_global_size(1);
    unsigned int nz = get_global_size(2);

    unsigned int idx = iz+nz*iy+ny*nz*ix;
    
    float3 dX[3];
    float3 dXb[3];
    float3 d2X[3][3];
    float3 Bx;
    float3 dB[3];
    float3 dBx[3];
    float Det_dX;
    float dDet_dX[3];
    
    unsigned int i;
    unsigned int j;
    unsigned int k;
    
    for (i = 0; i < 3; i++) {
        dX[i] = J[idx + i*nx*ny*nz];
        
        for (j = 0; j < 3; j++) {
            if (i == j) {
                Deriv(X, i, 2, F);
                d2X[i][j] = F[idx];
            } else {
                Deriv(&J[i*nx*ny*nz], j, 1, F);
                d2X[i][j] = F[idx];
            }  
        } 
    }
    
    for (i = 0; i < 3; i++) {
        Deriv(B, i, 1, F);
        dB[i] = F[idx];
    }
        
    Det_dX = dX[0].x*dX[1].y*dX[2].z + dX[0].y*dX[1].z*dX[2].x + dX[0].z*dX[1].x*dX[2].y - 
           dX[0].x*dX[1].z*dX[2].y - dX[0].y*dX[1].x*dX[2].z - dX[0].z*dX[1].y*dX[2].x;
    
    dXb[0] = Cross(dX[1], dX[2])/(float3)(Det_dX);
    dXb[1] = Cross(dX[2], dX[0])/(float3)(Det_dX);
    dXb[2] = Cross(dX[0], dX[1])/(float3)(Det_dX);
    
    Bx.x = dX[0].x*B[idx].x + dX[1].x*B[idx].y + dX[2].x*B[idx].z;
    Bx.y = dX[0].y*B[idx].x + dX[1].y*B[idx].y + dX[2].y*B[idx].z;
    Bx.z = dX[0].z*B[idx].x + dX[1].z*B[idx].y + dX[2].z*B[idx].z;
    
    Bx /= (float3)Det_dX;
    
    for (i = 0; i < 3; i++) {
        dDet_dX[i] = 0.;   
        for (j = 0; j < 3; j++) dDet_dX[i] += d2X[i][j].x*dXb[j].x + d2X[i][j].y*dXb[j].y + d2X[i][j].z*dXb[j].z;
        dDet_dX[i] *= Det_dX;
    }
       
    for (i = 0; i < 3; i++) {
        dBx[i] = -(float3)(dDet_dX[i])*Bx;

        dBx[i].x += dX[0].x*dB[i].x + dX[1].x*dB[i].y + dX[2].x*dB[i].z;
        dBx[i].y += dX[0].y*dB[i].x + dX[1].y*dB[i].y + dX[2].y*dB[i].z;
        dBx[i].z += dX[0].z*dB[i].x + dX[1].z*dB[i].y + dX[2].z*dB[i].z;
        
        dBx[i].x += d2X[0][i].x*B[idx].x + d2X[1][i].x*B[idx].y + d2X[2][i].x*B[idx].z;
        dBx[i].y += d2X[0][i].y*B[idx].x + d2X[1][i].y*B[idx].y + d2X[2][i].y*B[idx].z;
        dBx[i].z += d2X[0][i].z*B[idx].x + d2X[1][i].z*B[idx].y + d2X[2][i].z*B[idx].z;

        dBx[i] /= (float3)(Det_dX);
    }
    
    Ix[idx] = (float3)0;
    for (i = 0; i < 3; i++) {
        Ix[idx] += Cross(dXb[i], dBx[i]);
    }
    
    F[idx] = Cross(Ix[idx], Bx);
}



__kernel
void Step(__global float3* X, __global float3* F, __global float* params, float coeff)
{
    unsigned int ix = get_global_id(0);
    unsigned int iy = get_global_id(1);
    unsigned int iz = get_global_id(2);
    
    unsigned int nx = get_global_size(0);
    unsigned int ny = get_global_size(1);
    unsigned int nz = get_global_size(2);

    unsigned int idx = iz+nz*iy+ny*nz*ix;
    
//    if ((iz != 0) && (iz != (nz-1))) {    
    if ((ix != 0) && (ix != (nx-1)) && (iy != 0) && (iy != (ny-1)) && (iz != 0) && (iz != (nz-1))) {
        X[idx] += F[idx]*(float3)(params[0]*coeff);
    }
}



__kernel
void Update(__global float3* X, __global float3* Y, __global float* params)
{
    unsigned int ix = get_global_id(0);
    unsigned int iy = get_global_id(1);
    unsigned int iz = get_global_id(2);
    
    unsigned int nx = get_global_size(0);
    unsigned int ny = get_global_size(1);
    unsigned int nz = get_global_size(2);

    unsigned int idx = iz+nz*iy+ny*nz*ix;
    
    if (params[1] < 1.f) {
        X[idx] = Y[idx];   
    }
    
}


__kernel
void Error(__global float3* X, __global float3* Y, __global float* params)
{
    unsigned int ix = get_global_id(0);
    unsigned int iy = get_global_id(1);
    unsigned int iz = get_global_id(2);
    
    unsigned int nx = get_global_size(0);
    unsigned int ny = get_global_size(1);
    unsigned int nz = get_global_size(2);

    unsigned int idx = iz+nz*iy+ny*nz*ix;
    
    float err;
    
    err = ((X[idx].x-Y[idx].x)*(X[idx].x-Y[idx].x) + 
           (X[idx].y-Y[idx].y)*(X[idx].y-Y[idx].y) + 
           (X[idx].z-Y[idx].z)*(X[idx].z-Y[idx].z))/(params[2]*params[2]);
    
    if (err > params[1]) { 
        params[1] = err;
    }  
}


__kernel
void Colorize(__global float3* X, __global float4* C)
{
    unsigned int ix = get_global_id(0);
    unsigned int iy = get_global_id(1);
    unsigned int iz = get_global_id(2);
    
    unsigned int nx = get_global_size(0);
    unsigned int ny = get_global_size(1);
    unsigned int nz = get_global_size(2);

    unsigned int idx = iz+nz*iy+ny*nz*ix;
    
    float absX = sqrt(X[idx].x*X[idx].x + X[idx].y*X[idx].y + X[idx].z*X[idx].z);
    
    C[idx].x = absX*10.;
    C[idx].y = 1.-absX*10.;
    C[idx].z = 0.;
    C[idx].w = 1.;
    
}
