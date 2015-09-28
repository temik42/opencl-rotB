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
    unsigned int i = get_global_id(0);
    unsigned int j = get_global_id(1);
    unsigned int k = get_global_id(2);
    
    unsigned int nx = get_global_size(0);
    unsigned int ny = get_global_size(1);
    unsigned int nz = get_global_size(2);

    unsigned int idx = k+nz*j+ny*nz*i;
    
    unsigned int l;
    unsigned int n;
    unsigned int bias;
    
    if (d == 0) {
        n = nx;
        l = i;
        bias = ny*nz;
    }
    
    if (d == 1) {
        n = ny;
        l = j;
        bias = nz;
    }
    
    if (d == 2) {
        n = nz;
        l = k;
        bias = 1;
    }

    if ((l != 0) && (l != (n-1))) {
        if (m == 1) Y[idx] = X[idx+bias]*(float3)(0.5) - X[idx-bias]*(float3)(0.5);
        if (m == 2) Y[idx] = X[idx+bias] + X[idx-bias] - (float3)(2)*X[idx];       
    } else if (l == 0) {
        if (m == 1) Y[idx] = -(float3)(1.5)*X[idx] + (float3)(2)*X[idx+bias] - (float3)(0.5)*X[idx+2*bias]; 
        if (m == 2) Y[idx] = (float3)(2)*X[idx] - (float3)(5)*X[idx+bias] + (float3)(4)*X[idx+2*bias] - X[idx+3*bias];
    } else if (l == (n-1)) {
        if (m == 1) Y[idx] = (float3)(1.5)*X[idx] - (float3)(2)*X[idx-bias] + (float3)(0.5)*X[idx-2*bias];
        if (m == 2) Y[idx] = (float3)(2)*X[idx] - (float3)(5)*X[idx-bias] + (float3)(4)*X[idx-2*bias] - X[idx-3*bias];
    }
}

__kernel void flush(__global float* X, uchar idx)
{    
    X[idx] = (float)(0);    
}

__kernel void Flush(__global float3* X)
{
    unsigned int ix = get_global_id(0);
    unsigned int iy = get_global_id(1);
    unsigned int iz = get_global_id(2);
    
    unsigned int nx = get_global_size(0);
    unsigned int ny = get_global_size(1);
    unsigned int nz = get_global_size(2);

    unsigned int idx = iz+nz*iy+ny*nz*ix;
    
    X[idx] = (float3)(0);    
}


__kernel void Copy(__global float3* X, __global float3* Y)
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


__kernel void Jacobian(__global float3* X, __global float3* J)
{
    unsigned int ix = get_global_id(0);
    unsigned int iy = get_global_id(1);
    unsigned int iz = get_global_id(2);
    
    unsigned int nx = get_global_size(0);
    unsigned int ny = get_global_size(1);
    unsigned int nz = get_global_size(2);

    unsigned int idx = iz+nz*iy+ny*nz*ix;
    
    unsigned int i;
    for (i = 0; i < 3; i++) Deriv(X, i, 1, &J[i*nx*ny*nz]);    
}


__kernel void rotB(__global float3* X, __global float3* J, __global float3* B, __global float3* F)
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
    float3 dB[3];
    float3 Bx;
    float absBx;
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
    absBx = sqrt(Bx.x*Bx.x+Bx.y*Bx.x+Bx.z*Bx.z);
    
    //color[idx].x = Bx.x;
    //color[idx].x = (absBx-1.)*10.;
    //color[idx].y = Det_dX;
    //color[idx].z = Bx.z;
    
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
    

    F[idx].x = dBx[0].x*B[idx].x + dBx[1].x*B[idx].y + dBx[2].x*B[idx].z;
    F[idx].y = dBx[0].y*B[idx].x + dBx[1].y*B[idx].y + dBx[2].y*B[idx].z;
    F[idx].z = dBx[0].z*B[idx].x + dBx[1].z*B[idx].y + dBx[2].z*B[idx].z;
    
    F[idx] /= (float3)(Det_dX);
    
    for (i = 0; i < 3; i++) {
        F[idx].x -= Bx.x*dBx[i].x*dXb[i].x + Bx.y*dBx[i].y*dXb[i].x + Bx.z*dBx[i].z*dXb[i].x;
        F[idx].y -= Bx.x*dBx[i].x*dXb[i].y + Bx.y*dBx[i].y*dXb[i].y + Bx.z*dBx[i].z*dXb[i].y;
        F[idx].z -= Bx.x*dBx[i].x*dXb[i].z + Bx.y*dBx[i].y*dXb[i].z + Bx.z*dBx[i].z*dXb[i].z;
    }
    
    
    
    //F[idx].x += dDet_dX[0]/(Det_dX*Det_dX);
    //F[idx].y += dDet_dX[1]/(Det_dX*Det_dX);
    //F[idx].z += dDet_dX[2]/(Det_dX*Det_dX);
}


__kernel void Step(__global float3* X, __global float3* F, __global float* params, float coeff)
{
    unsigned int ix = get_global_id(0);
    unsigned int iy = get_global_id(1);
    unsigned int iz = get_global_id(2);
    
    unsigned int nx = get_global_size(0);
    unsigned int ny = get_global_size(1);
    unsigned int nz = get_global_size(2);

    unsigned int idx = iz+nz*iy+ny*nz*ix;
    
    if ((ix != 0) && (ix != (nx-1)) && (iy != 0) && (iy != (ny-1)) && (iz != 0) && (iz != (nz-1))) {
        X[idx] += F[idx]*(float3)(params[0]*coeff);
    }
}



__kernel void Update(__global float3* X, __global float3* Y, __global float* params)
{
    unsigned int ix = get_global_id(0);
    unsigned int iy = get_global_id(1);
    unsigned int iz = get_global_id(2);
    
    unsigned int nx = get_global_size(0);
    unsigned int ny = get_global_size(1);
    unsigned int nz = get_global_size(2);

    unsigned int idx = iz+nz*iy+ny*nz*ix;
    
    if (params[1] < 1.f) {
        if ((ix != 0) && (ix != (nx-1)) && (iy != 0) && (iy != (ny-1)) && (iz != 0) && (iz != (nz-1))) {
            X[idx] = Y[idx];   
        }
    }
    
}

__kernel void AdjustParams(__global float* params)
{
        params[0]/= pow(params[1], 0.1f);
        params[1] = 0.f;
}

__kernel void Error(__global float3* X, __global float3* Y, __global float3* F, __global float* params)
{
    unsigned int ix = get_global_id(0);
    unsigned int iy = get_global_id(1);
    unsigned int iz = get_global_id(2);
    
    unsigned int nx = get_global_size(0);
    unsigned int ny = get_global_size(1);
    unsigned int nz = get_global_size(2);

    unsigned int idx = iz+nz*iy+ny*nz*ix;
    
    float err;
    
    err = ((X[idx].x+F[idx].x-Y[idx].x)*(X[idx].x+F[idx].x-Y[idx].x) + 
                 (X[idx].y+F[idx].y-Y[idx].y)*(X[idx].y+F[idx].y-Y[idx].y) + 
                 (X[idx].z+F[idx].z-Y[idx].z)*(X[idx].z+F[idx].z-Y[idx].z))/(params[2]*params[2]);
    
    if (err > params[1]) { 
        params[1] = err;
    }
    
    
}

