#include <cstring>
#include <iostream>

#include "fhog.h"
#undef MIN

// platform independent aligned memory allocation (see also alFree)
void* alMalloc( size_t size, int alignment ) {
    const size_t pSize = sizeof(void*), a = alignment-1;
    void *raw = wrMalloc(size + a + pSize);
    void *aligned = (void*) (((size_t) raw + pSize + a) & ~a);
    *(void**) ((size_t) aligned-pSize) = raw;
    return aligned;
}

// platform independent alignned memory de-allocation (see also alMalloc)
void alFree(void* aligned) {
    void* raw = *(void**)((char*)aligned-sizeof(void*));
    wrFree(raw);
}

/*******************************************************************************
* Piotr's Computer Vision Matlab Toolbox      Version 3.30
* Copyright 2014 Piotr Dollar & Ron Appel.  [pdollar-at-gmail.com]
* Licensed under the Simplified BSD License [see external/bsd.txt]
*******************************************************************************/
// #include "wrappers.hpp"

#define PI 3.14159265f

// compute x and y gradients for just one column (uses sse)
void grad1( float *I, float *Gx, float *Gy, int h, int w, int x ) {
    int y, y1; float *Ip, *In, r; __m128 *_Ip, *_In, *_G, _r;
    // compute column of Gx
    Ip=I-h; In=I+h; r=.5f;
    if(x==0) { r=1; Ip+=h; } else if(x==w-1) { r=1; In-=h; }
    if( h<4 || h%4>0 || (size_t(I)&15) || (size_t(Gx)&15) ) {
        for( y=0; y<h; y++ ) *Gx++=(*In++-*Ip++)*r;
    } else {
        _G=(__m128*) Gx; _Ip=(__m128*) Ip; _In=(__m128*) In; _r = sse::SET(r);
        for(y=0; y<h; y+=4) *_G++=sse::MUL(sse::SUB(*_In++,*_Ip++),_r);
    }
    // compute column of Gy
#define GRADY(r) *Gy++=(*In++-*Ip++)*r;
    Ip=I; In=Ip+1;
    // GRADY(1); Ip--; for(y=1; y<h-1; y++) GRADY(.5f); In--; GRADY(1);
    y1=((~((size_t) Gy) + 1) & 15)/4; if(y1==0) y1=4; if(y1>h-1) y1=h-1;
    GRADY(1); Ip--; for(y=1; y<y1; y++) GRADY(.5f);
    _r = sse::SET(.5f); _G=(__m128*) Gy;
    for(; y+4<h-1; y+=4, Ip+=4, In+=4, Gy+=4)
        *_G++=sse::MUL(sse::SUB(sse::LDu(*In),sse::LDu(*Ip)),_r);
    for(; y<h-1; y++) GRADY(.5f); In--; GRADY(1);
#undef GRADY
}

// compute x and y gradients at each location (uses sse)
void grad2( float *I, float *Gx, float *Gy, int h, int w, int d ) {
    int o, x, c, a=w*h; for(c=0; c<d; c++) for(x=0; x<w; x++) {
        o=c*a+x*h; grad1( I+o, Gx+o, Gy+o, h, w, x );
    }
}

// build lookup table a[] s.t. a[x*n]~=acos(x) for x in [-1,1]
float* acosTable() {
    const int n=10000, b=10; int i;
    static float a[n*2+b*2]; static bool init=false;
    float *a1=a+n+b; if( init ) return a1;
    for( i=-n-b; i<-n; i++ )   a1[i]=PI;
    for( i=-n; i<n; i++ )      a1[i]=float(acos(i/float(n)));
    for( i=n; i<n+b; i++ )     a1[i]=0;
    for( i=-n-b; i<n/10; i++ ) if( a1[i] > PI-1e-6f ) a1[i]=PI-1e-6f;
    init=true; return a1;
}

// compute gradient magnitude and orientation at each location (uses sse)
void gradMag( float *I, float *M, float *O, int h, int w, int d, bool full ) {
    int y;
    __m128 *_Gx, *_Gy, *_M2, _m;
    float *acost = acosTable(), acMult=10000.0f;
    // allocate memory for storing one column of output (padded so h4%4==0)
    int h4=(h%4==0) ? h : h-(h%4)+4;
    int s = static_cast<size_t>(d) * static_cast<size_t>(h4) * sizeof(float);
    float* M2=(float*) alMalloc(s,16); _M2=(__m128*) M2;
    float* Gx=(float*) alMalloc(s,16); _Gx=(__m128*) Gx;
    float* Gy=(float*) alMalloc(s,16); _Gy=(__m128*) Gy;
    // compute gradient magnitude and orientation for each column
    for(int x=0; x<w; x++ )
    {
        // compute gradients (Gx, Gy) with maximum squared magnitude (M2)
        for(int c=0; c<d; c++)
        {
            grad1( I+x*h+c*w*h, Gx+c*h4, Gy+c*h4, h, w, x );
            for( y=0; y<h4/4; y++ ) {
                int y1=h4/4*c+y;
                _M2[y1]=sse::ADD(sse::MUL(_Gx[y1],_Gx[y1]),sse::MUL(_Gy[y1],_Gy[y1]));
                if( c==0 )
                    continue;
                _m = sse::CMPGT( _M2[y1], _M2[y] );
                _M2[y] = sse::OR( sse::AND(_m,_M2[y1]), sse::ANDNOT(_m,_M2[y]) );
                _Gx[y] = sse::OR( sse::AND(_m,_Gx[y1]), sse::ANDNOT(_m,_Gx[y]) );
                _Gy[y] = sse::OR( sse::AND(_m,_Gy[y1]), sse::ANDNOT(_m,_Gy[y]) );
            }
        }
        // compute gradient mangitude (M) and normalize Gx
        for( y=0; y<h4/4; y++ ) {
            _m = sse::MIN( sse::RCPSQRT(_M2[y]), sse::SET(1e10f) );
            _M2[y] = sse::RCP(_m);
            if(O) _Gx[y] = sse::MUL( sse::MUL(_Gx[y],_m), sse::SET(acMult) );
            if(O) _Gx[y] = sse::XOR( _Gx[y], sse::AND(_Gy[y], sse::SET(-0.f)) );
        };
        memcpy( M+x*h, M2, h*sizeof(float) );
        // compute and store gradient orientation (O) via table lookup
        if( O!=0 ) for( y=0; y<h; y++ ) O[x*h+y] = acost[(int)Gx[y]];
        if( O!=0 && full ) {
            int y1=((~size_t(O+x*h)+1)&15)/4; y=0;
            for( ; y<y1; y++ ) O[y+x*h]+=(Gy[y]<0)*PI;
            for( ; y<h-4; y+=4 ) sse::STRu( O[y+x*h],
                    sse::ADD( sse::LDu(O[y+x*h]), sse::AND(sse::CMPLT(sse::LDu(Gy[y]),sse::SET(0.f)),sse::SET(PI)) ) );
            for( ; y<h; y++ ) O[y+x*h]+=(Gy[y]<0)*PI;
        }
    }
    alFree(Gx); alFree(Gy); alFree(M2);
}

// normalize gradient magnitude at each location (uses sse)
void gradMagNorm( float *M, float *S, int h, int w, float norm ) {
    __m128 *_M, *_S, _norm; int i=0, n=h*w, n4=n/4;
    _S = (__m128*) S; _M = (__m128*) M; _norm = sse::SET(norm);
    bool sse = !(size_t(M)&15) && !(size_t(S)&15);
    if(sse)
        for(; i<n4; i++)
        {
            *_M=sse::MUL(*_M,sse::RCP(sse::ADD(*_S++,_norm))); _M++;
        }
    if(sse)
        i*=4;
    for(; i<n; i++) M[i] /= (S[i] + norm);
}

// helper for gradHist, quantize O and M into O0, O1 and M0, M1 (uses sse)
void gradQuantize( float *O, float *M, int *O0, int *O1, float *M0, float *M1,
                   int nb, int n, float norm, int nOrients, bool full, bool interpolate )
{
    // assumes all *OUTPUT* matrices are 4-byte aligned
    int i, o0, o1; float o, od, m;
    __m128i _o0, _o1, *_O0, *_O1; __m128 _o, _od, _m, *_M0, *_M1;
    // define useful constants
    const float oMult=(float)nOrients/(full?2*PI:PI); const int oMax=nOrients*nb;
    const __m128 _norm=sse::SET(norm), _oMult=sse::SET(oMult), _nbf=sse::SET((float)nb);
    const __m128i _oMax=sse::SET(oMax), _nb=sse::SET(nb);
    // perform the majority of the work with sse
    _O0=(__m128i*) O0; _O1=(__m128i*) O1; _M0=(__m128*) M0; _M1=(__m128*) M1;
    if( interpolate ) for( i=0; i<=n-4; i+=4 ) {
        _o=sse::MUL(sse::LDu(O[i]),_oMult); _o0=sse::CVT(_o); _od=sse::SUB(_o,sse::CVT(_o0));
        _o0=sse::CVT(sse::MUL(sse::CVT(_o0),_nbf)); _o0=sse::AND(sse::CMPGT(_oMax,_o0),_o0); *_O0++=_o0;
        _o1=sse::ADD(_o0,_nb); _o1=sse::AND(sse::CMPGT(_oMax,_o1),_o1); *_O1++=_o1;
        _m=sse::MUL(sse::LDu(M[i]),_norm); *_M1=sse::MUL(_od,_m); *_M0++=sse::SUB(_m,*_M1); _M1++;
    } else for( i=0; i<=n-4; i+=4 ) {
        _o=sse::MUL(sse::LDu(O[i]),_oMult); _o0=sse::CVT(sse::ADD(_o,sse::SET(.5f)));
        _o0=sse::CVT(sse::MUL(sse::CVT(_o0),_nbf)); _o0=sse::AND(sse::CMPGT(_oMax,_o0),_o0); *_O0++=_o0;
        *_M0++=sse::MUL(sse::LDu(M[i]),_norm); *_M1++=sse::SET(0.f); *_O1++=sse::SET(0);
    }
    // compute trailing locations without sse
    if( interpolate ) for(; i<n; i++ ) {
        o=O[i]*oMult; o0=(int) o; od=o-o0;
        o0*=nb; if(o0>=oMax) o0=0; O0[i]=o0;
        o1=o0+nb; if(o1==oMax) o1=0; O1[i]=o1;
        m=M[i]*norm; M1[i]=od*m; M0[i]=m-M1[i];
    } else for(; i<n; i++ ) {
        o=O[i]*oMult; o0=(int) (o+.5f);
        o0*=nb; if(o0>=oMax) o0=0; O0[i]=o0;
        M0[i]=M[i]*norm; M1[i]=0; O1[i]=0;
    }
}

// compute nOrients gradient histograms per bin x bin block of pixels
void gradHist( float *M, float *O, float *H, int h, int w,
               int bin, int nOrients, int softBin, bool full )
{
    const int hb=h/bin, wb=w/bin, h0=hb*bin, w0=wb*bin, nb=wb*hb;
    const float s=(float)bin, sInv=1/s, sInv2=1/s/s;
    float *H0, *H1, *M0, *M1; int x, y; int *O0, *O1; float xb = 0, init = 0;
    O0=(int*)alMalloc(h*sizeof(int),16); M0=(float*) alMalloc(h*sizeof(float),16);
    O1=(int*)alMalloc(h*sizeof(int),16); M1=(float*) alMalloc(h*sizeof(float),16);
    // main loop
    for( x=0; x<w0; x++ ) {
        // compute target orientation bins for entire column - very fast
        gradQuantize(O+x*h,M+x*h,O0,O1,M0,M1,nb,h0,sInv2,nOrients,full,softBin>=0);

        if( softBin<0 && softBin%2==0 ) {
            // no interpolation w.r.t. either orienation or spatial bin
            H1=H+(x/bin)*hb;
#define GH H1[O0[y]]+=M0[y]; y++;
            if( bin==1 )      for(y=0; y<h0;) { GH; H1++; }
            else if( bin==2 ) for(y=0; y<h0;) { GH; GH; H1++; }
            else if( bin==3 ) for(y=0; y<h0;) { GH; GH; GH; H1++; }
            else if( bin==4 ) for(y=0; y<h0;) { GH; GH; GH; GH; H1++; }
            else for( y=0; y<h0;) { for( int y1=0; y1<bin; y1++ ) { GH; } H1++; }
#undef GH

        } else if( softBin%2==0 || bin==1 ) {
            // interpolate w.r.t. orientation only, not spatial bin
            H1=H+(x/bin)*hb;
#define GH H1[O0[y]]+=M0[y]; H1[O1[y]]+=M1[y]; y++;
            if( bin==1 )      for(y=0; y<h0;) { GH; H1++; }
            else if( bin==2 ) for(y=0; y<h0;) { GH; GH; H1++; }
            else if( bin==3 ) for(y=0; y<h0;) { GH; GH; GH; H1++; }
            else if( bin==4 ) for(y=0; y<h0;) { GH; GH; GH; GH; H1++; }
            else for( y=0; y<h0;) { for( int y1=0; y1<bin; y1++ ) { GH; } H1++; }
#undef GH

        } else {
            // interpolate using trilinear interpolation
            float ms[4], xyd, yb, xd, yd; __m128 _m, _m0, _m1;
			if (x == 0) { init = (0 + .5f)*sInv - 0.5f; xb = init; }
            bool hasLf = xb>=0;
			int xb0 = hasLf?(int)xb:-1;
			bool hasRt = xb0 < wb-1;
            xd=xb-xb0;
			xb+=sInv;
			yb=init;
			y=0;
			int yb0 = -1;
            // macros for code conciseness
#define GHinit yd=yb-yb0; yb+=sInv; H0=H+xb0*hb+yb0; xyd=xd*yd; \
    ms[0]=1-xd-yd+xyd; ms[1]=yd-xyd; ms[2]=xd-xyd; ms[3]=xyd;
#define GH(H,ma,mb) H1=H; sse::STRu(*H1,sse::ADD(sse::LDu(*H1),sse::MUL(ma,mb)));
            // leading rows, no top bin
            for( ; y<bin/2; y++ ) {
                yb0=-1; GHinit;
                if(hasLf) { H0[O0[y]+1]+=ms[1]*M0[y]; H0[O1[y]+1]+=ms[1]*M1[y]; }
                if(hasRt) { H0[O0[y]+hb+1]+=ms[3]*M0[y]; H0[O1[y]+hb+1]+=ms[3]*M1[y]; }
            }
            // main rows, has top and bottom bins, use SSE for minor speedup
            if( softBin<0 ) for( ; ; y++ ) {
                yb0 = (int) yb; if(yb0>=hb-1) break; GHinit; _m0=sse::SET(M0[y]);
                if(hasLf) { _m=sse::SET(0,0,ms[1],ms[0]); GH(H0+O0[y],_m,_m0); }
                if(hasRt) { _m=sse::SET(0,0,ms[3],ms[2]); GH(H0+O0[y]+hb,_m,_m0); }
            } else for( ; ; y++ ) {
                yb0 = (int) yb; if(yb0>=hb-1) break; GHinit;
                _m0=sse::SET(M0[y]); _m1=sse::SET(M1[y]);
                if(hasLf) { _m=sse::SET(0,0,ms[1],ms[0]);
                    GH(H0+O0[y],_m,_m0); GH(H0+O1[y],_m,_m1); }
                if(hasRt) { _m=sse::SET(0,0,ms[3],ms[2]);
                    GH(H0+O0[y]+hb,_m,_m0); GH(H0+O1[y]+hb,_m,_m1); }
            }
            // final rows, no bottom bin
            for( ; y<h0; y++ ) {
                yb0 = (int) yb; GHinit;
                if(hasLf) { H0[O0[y]]+=ms[0]*M0[y]; H0[O1[y]]+=ms[0]*M1[y]; }
                if(hasRt) { H0[O0[y]+hb]+=ms[2]*M0[y]; H0[O1[y]+hb]+=ms[2]*M1[y]; }
            }
#undef GHinit
#undef GH
        }
    }
    alFree(O0); alFree(O1); alFree(M0); alFree(M1);
    // normalize boundary bins which only get 7/8 of weight of interior bins
    if( softBin%2!=0 ) for( int o=0; o<nOrients; o++ ) {
        x=0; for( y=0; y<hb; y++ ) H[o*nb+x*hb+y]*=8.f/7.f;
        y=0; for( x=0; x<wb; x++ ) H[o*nb+x*hb+y]*=8.f/7.f;
        x=wb-1; for( y=0; y<hb; y++ ) H[o*nb+x*hb+y]*=8.f/7.f;
        y=hb-1; for( x=0; x<wb; x++ ) H[o*nb+x*hb+y]*=8.f/7.f;
    }
}

/******************************************************************************/

// HOG helper: compute 2x2 block normalization values (padded by 1 pixel)
float* hogNormMatrix( float *H, int nOrients, int hb, int wb, int bin ) {
    int o, x, y, dx, dy, hb1=hb+1, wb1=wb+1;
    float eps = 1e-4f/4/bin/bin/bin/bin; // precise backward equality
    float* N = (float*) wrCalloc(static_cast<size_t>(hb1) * static_cast<size_t>(wb1), sizeof(float));
    float* N1=N+hb1+1;
    for( o=0; o<nOrients; o++ ) for( x=0; x<wb; x++ ) for( y=0; y<hb; y++ )
        N1[x*hb1+y] += H[o*wb*hb+x*hb+y]*H[o*wb*hb+x*hb+y];
    for( x=0; x<wb-1; x++ ) for( y=0; y<hb-1; y++ ) {
        float* n=N1+x*hb1+y; *n=1/float(sqrt(n[0]+n[1]+n[hb1]+n[hb1+1]+eps)); }
    x=0;     dx= 1; dy= 1; y=0;                  N[x*hb1+y]=N[(x+dx)*hb1+y+dy];
    x=0;     dx= 1; dy= 0; for(y=0; y<hb1; y++)  N[x*hb1+y]=N[(x+dx)*hb1+y+dy];
    x=0;     dx= 1; dy=-1; y=hb1-1;              N[x*hb1+y]=N[(x+dx)*hb1+y+dy];
    x=wb1-1; dx=-1; dy= 1; y=0;                  N[x*hb1+y]=N[(x+dx)*hb1+y+dy];
    x=wb1-1; dx=-1; dy= 0; for( y=0; y<hb1; y++) N[x*hb1+y]=N[(x+dx)*hb1+y+dy];
    x=wb1-1; dx=-1; dy=-1; y=hb1-1;              N[x*hb1+y]=N[(x+dx)*hb1+y+dy];
    y=0;     dx= 0; dy= 1; for(x=0; x<wb1; x++)  N[x*hb1+y]=N[(x+dx)*hb1+y+dy];
    y=hb1-1; dx= 0; dy=-1; for(x=0; x<wb1; x++)  N[x*hb1+y]=N[(x+dx)*hb1+y+dy];
    return N;
}

// HOG helper: compute HOG or FHOG channels
void hogChannels( float *H, const float *R, const float *N,
                  int hb, int wb, int nOrients, float clip, int type )
{
#define GETT(blk) t=R1[y]*N1[y-(blk)]; if(t>clip) t=clip; c++;
    const float r=.2357f; int o, x, y, c; float t;
    const int nb=wb*hb, nbo=nOrients*nb, hb1=hb+1;
    for( o=0; o<nOrients; o++ ) for( x=0; x<wb; x++ ) {
        const float *R1=R+o*nb+x*hb, *N1=N+x*hb1+hb1+1;
        float *H1 = (type<=1) ? (H+o*nb+x*hb) : (H+x*hb);
        if( type==0) for( y=0; y<hb; y++ ) {
            // store each orientation and normalization (nOrients*4 channels)
            c=-1; GETT(0); H1[c*nbo+y]=t; GETT(1); H1[c*nbo+y]=t;
            GETT(hb1); H1[c*nbo+y]=t; GETT(hb1+1); H1[c*nbo+y]=t;
        } else if( type==1 ) for( y=0; y<hb; y++ ) {
            // sum across all normalizations (nOrients channels)
            c=-1; GETT(0); H1[y]+=t*.5f; GETT(1); H1[y]+=t*.5f;
            GETT(hb1); H1[y]+=t*.5f; GETT(hb1+1); H1[y]+=t*.5f;
        } else if( type==2 ) for( y=0; y<hb; y++ ) {
            // sum across all orientations (4 channels)
            c=-1; GETT(0); H1[c*nb+y]+=t*r; GETT(1); H1[c*nb+y]+=t*r;
            GETT(hb1); H1[c*nb+y]+=t*r; GETT(hb1+1); H1[c*nb+y]+=t*r;
        }
    }
#undef GETT
}

// compute HOG features
void hog( float *M, float *O, float *H, int h, int w, int binSize,
          int nOrients, int softBin, bool full, float clip )
{
    //float *N, *R; const int hb=h/binSize, wb=w/binSize, nb=hb*wb;
    float *N, *R; const int hb=h/binSize, wb=w/binSize;
    // compute unnormalized gradient histograms
    R = (float*) wrCalloc(wb*hb*nOrients+2,sizeof(float));
    gradHist( M, O, R, h, w, binSize, nOrients, softBin, full );
    // compute block normalization values
    N = hogNormMatrix( R, nOrients, hb, wb, binSize );
    // perform four normalizations per spatial block
    hogChannels( H, R, N, hb, wb, nOrients, clip, 0 );
    wrFree(N); wrFree(R);
}

// compute FHOG features
void fhog( float *M, float *O, float *H, int h, int w, int binSize,
           int nOrients, int softBin, float clip )
{
    const int hb=h/binSize, wb=w/binSize, nb=hb*wb, nbo=nb*nOrients;
    // compute unnormalized constrast sensitive histograms
    float* R1 = (float*) wrCalloc(static_cast<size_t>(wb) * static_cast<size_t>(hb) * static_cast<size_t>(nOrients) * 2 + 2, sizeof(float));
    gradHist( M, O, R1, h, w, binSize, nOrients*2, softBin, true );
    // compute unnormalized contrast insensitive histograms
    float* R2 = (float*) wrCalloc(static_cast<size_t>(wb) * static_cast<size_t>(hb) * static_cast<size_t>(nOrients), sizeof(float));
    for(int o=0; o<nOrients; o++ ) for(int x=0; x<nb; x++ )
        R2[o*nb+x] = R1[o*nb+x]+R1[(o+nOrients)*nb+x];
    // compute block normalization values
    float* N = hogNormMatrix( R2, nOrients, hb, wb, binSize );
    // normalized histograms and texture channels
    hogChannels( H+nbo*0, R1, N, hb, wb, nOrients*2, clip, 1 );
    hogChannels( H+nbo*2, R2, N, hb, wb, nOrients*1, clip, 1 );
    hogChannels( H+nbo*3, R1, N, hb, wb, nOrients*2, clip, 2 );
    wrFree(N); wrFree(R1); wrFree(R2);
}

/******************************************************************************/
#ifdef MATLAB_MEX_FILE
// Create [hxwxd] mxArray array, initialize to 0 if c=true
mxArray* mxCreateMatrix3( int h, int w, int d, mxClassID id, bool c, void **I ){
    const int dims[3]={h,w,d}, n=h*w*d; int b; mxArray* M;
    if( id==mxINT32_CLASS ) b=sizeof(int);
    else if( id==mxDOUBLE_CLASS ) b=sizeof(double);
    else if( id==mxSINGLE_CLASS ) b=sizeof(float);
    else mexErrMsgTxt("Unknown mxClassID.");
    *I = c ? mxCalloc(n,b) : mxMalloc(n*b);
    M = mxCreateNumericMatrix(0,0,id,mxREAL);
    mxSetData(M,*I); mxSetDimensions(M,dims,3); return M;
}

// Check inputs and outputs to mex, retrieve first input I
void checkArgs( int nl, mxArray *pl[], int nr, const mxArray *pr[], int nl0,
                int nl1, int nr0, int nr1, int *h, int *w, int *d, mxClassID id, void **I )
{
    const int *dims; int nDims;
    if( nl<nl0 || nl>nl1 ) mexErrMsgTxt("Incorrect number of outputs.");
    if( nr<nr0 || nr>nr1 ) mexErrMsgTxt("Incorrect number of inputs.");
    nDims = mxGetNumberOfDimensions(pr[0]); dims = mxGetDimensions(pr[0]);
    *h=dims[0]; *w=dims[1]; *d=(nDims==2) ? 1 : dims[2]; *I = mxGetPr(pr[0]);
    if( nDims!=2 && nDims!=3 ) mexErrMsgTxt("I must be a 2D or 3D array.");
    if( mxGetClassID(pr[0])!=id ) mexErrMsgTxt("I has incorrect type.");
}

// [Gx,Gy] = grad2(I) - see gradient2.m
void mGrad2( int nl, mxArray *pl[], int nr, const mxArray *pr[] ) {
    int h, w, d; float *I, *Gx, *Gy;
    checkArgs(nl,pl,nr,pr,1,2,1,1,&h,&w,&d,mxSINGLE_CLASS,(void**)&I);
    if(h<2 || w<2) mexErrMsgTxt("I must be at least 2x2.");
    pl[0]= mxCreateMatrix3( h, w, d, mxSINGLE_CLASS, 0, (void**) &Gx );
    pl[1]= mxCreateMatrix3( h, w, d, mxSINGLE_CLASS, 0, (void**) &Gy );
    grad2( I, Gx, Gy, h, w, d );
}

// [M,O] = gradMag( I, channel, full ) - see gradientMag.m
void mGradMag( int nl, mxArray *pl[], int nr, const mxArray *pr[] ) {
    int h, w, d, c, full; float *I, *M, *O=0;
    checkArgs(nl,pl,nr,pr,1,2,3,3,&h,&w,&d,mxSINGLE_CLASS,(void**)&I);
    if(h<2 || w<2) mexErrMsgTxt("I must be at least 2x2.");
    c = (int) mxGetScalar(pr[1]); full = (int) mxGetScalar(pr[2]);
    if( c>0 && c<=d ) { I += h*w*(c-1); d=1; }
    pl[0] = mxCreateMatrix3(h,w,1,mxSINGLE_CLASS,0,(void**)&M);
    if(nl>=2) pl[1] = mxCreateMatrix3(h,w,1,mxSINGLE_CLASS,0,(void**)&O);
    gradMag(I, M, O, h, w, d, full>0 );
}

// gradMagNorm( M, S, norm ) - operates on M - see gradientMag.m
void mGradMagNorm( int nl, mxArray *pl[], int nr, const mxArray *pr[] ) {
    int h, w, d; float *M, *S, norm;
    checkArgs(nl,pl,nr,pr,0,0,3,3,&h,&w,&d,mxSINGLE_CLASS,(void**)&M);
    if( mxGetM(pr[1])!=h || mxGetN(pr[1])!=w || d!=1 ||
            mxGetClassID(pr[1])!=mxSINGLE_CLASS ) mexErrMsgTxt("M or S is bad.");
    S = (float*) mxGetPr(pr[1]); norm = (float) mxGetScalar(pr[2]);
    gradMagNorm(M,S,h,w,norm);
}

// H=gradHist(M,O,[...]) - see gradientHist.m
void mGradHist( int nl, mxArray *pl[], int nr, const mxArray *pr[] ) {
    int h, w, d, hb, wb, nChns, binSize, nOrients, softBin, useHog;
    bool full; float *M, *O, *H, clipHog;
    checkArgs(nl,pl,nr,pr,1,3,2,8,&h,&w,&d,mxSINGLE_CLASS,(void**)&M);
    O = (float*) mxGetPr(pr[1]);
    if( mxGetM(pr[1])!=h || mxGetN(pr[1])!=w || d!=1 ||
            mxGetClassID(pr[1])!=mxSINGLE_CLASS ) mexErrMsgTxt("M or O is bad.");
    binSize  = (nr>=3) ? (int)   mxGetScalar(pr[2])    : 8;
    nOrients = (nr>=4) ? (int)   mxGetScalar(pr[3])    : 9;
    softBin  = (nr>=5) ? (int)   mxGetScalar(pr[4])    : 1;
    useHog   = (nr>=6) ? (int)   mxGetScalar(pr[5])    : 0;
    clipHog  = (nr>=7) ? (float) mxGetScalar(pr[6])    : 0.2f;
    full     = (nr>=8) ? (bool) (mxGetScalar(pr[7])>0) : false;
    hb = h/binSize; wb = w/binSize;
    nChns = useHog== 0 ? nOrients : (useHog==1 ? nOrients*4 : nOrients*3+5);
    pl[0] = mxCreateMatrix3(hb,wb,nChns,mxSINGLE_CLASS,1,(void**)&H);
    if( nOrients==0 ) return;
    if( useHog==0 ) {
        gradHist( M, O, H, h, w, binSize, nOrients, softBin, full );
    } else if(useHog==1) {
        hog( M, O, H, h, w, binSize, nOrients, softBin, full, clipHog );
    } else {
        fhog( M, O, H, h, w, binSize, nOrients, softBin, clipHog );
    }
}

// inteface to various gradient functions (see corresponding Matlab functions)
void mexFunction( int nl, mxArray *pl[], int nr, const mxArray *pr[] ) {
    int f; char action[1024]; f=mxGetString(pr[0],action,1024); nr--; pr++;
    if(f) mexErrMsgTxt("Failed to get action.");
    else if(!strcmp(action,"gradient2")) mGrad2(nl,pl,nr,pr);
    else if(!strcmp(action,"gradientMag")) mGradMag(nl,pl,nr,pr);
    else if(!strcmp(action,"gradientMagNorm")) mGradMagNorm(nl,pl,nr,pr);
    else if(!strcmp(action,"gradientHist")) mGradHist(nl,pl,nr,pr);
    else mexErrMsgTxt("Invalid action.");
}
#endif


float* crop_H(float *H,int* h_height,int* h_width,int depth,int dh,int dw) {
    int crop_h = *h_height-dh-1;
    int crop_w = *h_width-dw-1;
    float* crop_H = new float[crop_h*crop_w*depth];

    for(int i = 1;i < *h_height-dh;i ++)
        for(int j = 1;j < *h_width-dw;j ++)
            for(int k = 0;k < depth;k ++)
                crop_H[i-1 + (j-1)*(crop_h) + k*(crop_h*crop_w)] = H[k*(*h_width * *h_height) + j*(*h_height) + i];
    delete []H;
    *h_height = crop_h;*h_width = crop_w;
    return crop_H;
}

float* fhog(float *M,float* O,int height,int width,int /*channel*/,int *h,int *w,int *d,int binSize, int nOrients, float clip, bool crop) {
    *h = height/binSize;
    *w = width/binSize;
    *d = nOrients*3+5;
    const size_t allSize = static_cast<size_t>(*h) * static_cast<size_t>(*w) * static_cast<size_t>(*d);

    float* H = new float[allSize];
    memset(H, 0, allSize * sizeof(float));

    fhog( M, O, H, height, width, binSize, nOrients, -1, clip );

    if(!crop)
        return H;
    return crop_H(H,h,w,*d,height%binSize < binSize/2,width%binSize < binSize/2);
}

void fhog(cv::MatND &fhog_feature, const cv::Mat& input, int binSize, int nOrients, float clip, bool crop) {
    int HEIGHT = input.rows;
    int WIDTH = input.cols;
    int DEPTH = input.channels();

    float *II = new float[HEIGHT*WIDTH*DEPTH];
    int count=0;

    // MatLab:: RGB, OpenCV: BGR

    for (int i = 0; i < WIDTH; i++) {
        for (int j = 0; j < HEIGHT; j++) {
            cv::Vec3b p = input.at<cv::Vec3b>(j,i);
            II[count+2] = p[0]; // B->R
            II[count+1] = p[1]; // G->G
            II[count+0] = p[2]; // R->B
            count += 3;
        }
    }

    float *I = new float[HEIGHT*WIDTH*DEPTH];

    // channel x width x height
    for (int i = 0; i < WIDTH; i++) {
        for (int j = 0; j < HEIGHT; j++) {
            for (int k = 0; k < DEPTH; k++) {
                I[k*WIDTH*HEIGHT+i*HEIGHT+j] = II[i*HEIGHT*DEPTH+j*DEPTH+k];
            }
        }
    }

    float *M = new float[HEIGHT*WIDTH], *O = new float[HEIGHT*WIDTH];
    gradMag(I, M, O, HEIGHT, WIDTH, DEPTH, true);

    int h,w,d;
    float* HH = fhog(M,O,HEIGHT,WIDTH,DEPTH,&h,&w,&d,binSize,nOrients,clip,crop);
    float* H = new float[w*h*d];

    for(int i = 0;i < w; i++)
        for(int j = 0;j < h; j++)
            for(int k = 0;k < d; k++)
                //H[i*h*d+j*d+k] = HH[k*w*h+i*h+j]; // ->hwd
                H[j*w*d+i*d+k] = HH[k*w*h+i*h+j]; // ->whd

    fhog_feature = cv::MatND(h,w,CV_32FC(32),H).clone();

    delete[] H;

    delete[] M; delete[] O;
    delete[] II;delete[] I;delete[] HH;
}

void fhog28(cv::MatND &fhog_feature, const cv::Mat& input, int binSize, int nOrients, float clip, bool crop) {
    int HEIGHT = input.rows;
    int WIDTH = input.cols;
    int DEPTH = input.channels();

    float *II = new float[WIDTH*HEIGHT*DEPTH];
    int count=0;

    // MatLab:: RGB, OpenCV: BGR

    for (int i = 0; i < WIDTH; i++) {
        for (int j = 0; j < HEIGHT; j++) {
            cv::Vec3b p = input.at<cv::Vec3b>(j,i);
            II[count+2] = p[0]; // B->R
            II[count+1] = p[1]; // G->G
            II[count+0] = p[2]; // R->B
            count += 3;
        }
    }

    float *I = new float[HEIGHT*WIDTH*DEPTH];

    // channel x width x height
    for (int i = 0; i < WIDTH; i++) {
        for (int j = 0; j < HEIGHT; j++) {
            for (int k = 0; k < DEPTH; k++) {
                I[k*WIDTH*HEIGHT+i*HEIGHT+j] = II[i*HEIGHT*DEPTH+j*DEPTH+k];
            }
        }
    }

    float *M = new float[HEIGHT*WIDTH], *O = new float[HEIGHT*WIDTH];
    gradMag(I, M, O, HEIGHT, WIDTH, DEPTH, true);

    int h,w,d;
    float* HH = fhog(M,O,HEIGHT,WIDTH,DEPTH,&h,&w,&d,binSize,nOrients,clip,crop);

#undef CHANNELS
#define CHANNELS 28

    assert(d >= CHANNELS);

    // out = zeros(h, w, 28, 'single');
    // out(:,:,2:28) = temp(:,:,1:27);

    float* H = new float[w*h*CHANNELS];

    for(int i = 0;i < w; i++)
        for(int j = 0;j < h; j++) {
            //H[i*h*CHANNELS+j*CHANNELS+0] = 0.0;
            H[j*w*CHANNELS+i*CHANNELS+0] = 0.0;
            for(int k = 0;k < CHANNELS-1;k++) {
                //H[i*h*CHANNELS+j*CHANNELS+k+1] = HH[k*w*h+i*h+j]; // ->hwd
                H[j*w*CHANNELS+i*CHANNELS+k+1] = HH[k*w*h+i*h+j]; // ->whd
            }
        }

    fhog_feature = cv::MatND(h,w,CV_32FC(CHANNELS),H).clone();

    delete[] H;

    delete[] M; delete[] O;
    delete[] II;delete[] I;delete[] HH;
}

void fhog31(cv::MatND &fhog_feature, const cv::Mat& input, int binSize, int nOrients, float clip, bool crop) {
    int HEIGHT = input.rows;
    int WIDTH = input.cols;
    int DEPTH = input.channels();

    float *II = new float[WIDTH*HEIGHT*DEPTH];
    int count=0;

    // MatLab:: RGB, OpenCV: BGR

    for (int i = 0; i < WIDTH; i++) {
        for (int j = 0; j < HEIGHT; j++) {
            cv::Vec3b p = input.at<cv::Vec3b>(j,i);
            II[count+2] = p[0]; // B->R
            II[count+1] = p[1]; // G->G
            II[count+0] = p[2]; // R->B
            count += 3;
        }
    }

    float *I = new float[HEIGHT*WIDTH*DEPTH];

    // channel x width x height
    for (int i = 0; i < WIDTH; i++) {
        for (int j = 0; j < HEIGHT; j++) {
            for (int k = 0; k < DEPTH; k++) {
                I[k*WIDTH*HEIGHT+i*HEIGHT+j] = II[i*HEIGHT*DEPTH+j*DEPTH+k];
            }
        }
    }

    float *M = new float[HEIGHT*WIDTH], *O = new float[HEIGHT*WIDTH];
    gradMag(I, M, O, HEIGHT, WIDTH, DEPTH, true);

    int h,w,d;
    float* HH = fhog(M,O,HEIGHT,WIDTH,DEPTH,&h,&w,&d,binSize,nOrients,clip,crop);

#undef CHANNELS
#define CHANNELS 31

    assert(d >= CHANNELS);

    // out = zeros(h, w, 31, 'single');
    // out(:,:,1:31) = temp(:,:,1:31);

    float* H = new float[w*h*CHANNELS];

    for(int i = 0;i < w; i++)
        for(int j = 0;j < h; j++) {
            for(int k = 0;k < CHANNELS;k++) {
                //H[i*h*CHANNELS+j*CHANNELS+k+1] = HH[k*w*h+i*h+j]; // ->hwd
                H[j*w*CHANNELS+i*CHANNELS+k] = HH[k*w*h+i*h+j]; // ->whd
            }
        }

    fhog_feature = cv::MatND(h,w,CV_32FC(CHANNELS),H).clone();

    delete[] H;

    delete[] M; delete[] O;
    delete[] II;delete[] I;delete[] HH;
}
