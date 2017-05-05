#include <iostream>
#include <fstream>
#include <algorithm>
#include <numeric>
#include <cassert>
#include <string>
#include <cmath>
#include <vector>

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#define USE_DOUBLE

#ifdef USE_DOUBLE
typedef double REAL;
#else
typedef float REAL;
#endif

template<class T> class Array2dC;

template<class T>
class Array2d
{
public:
    int nrow;
    int ncol;
    T** p;
public:
    Array2d():nrow(0),ncol(0),p(NULL) { }
    Array2d(const int nrow,const int ncol):nrow(0),ncol(0),p(NULL)
    {
        Create(nrow,ncol);
    }
    Array2d(const Array2d<T>& source);
    virtual ~Array2d()
    {
        Clear();
    }

    Array2d<T>& operator=(const Array2d<T>& source);
    void Create(const int _nrow,const int _ncol);
    void Swap(Array2d<T>& array2);
    void Clear();
    void Zero(const T t = 0);
};

template<class T>
class Array2dC
{
public:
    int nrow;
    int ncol;
    T** p;
    T* buf;
public:
    Array2dC():nrow(0),ncol(0),p(NULL),buf(NULL) {}
    Array2dC(const int nrow,const int ncol):nrow(0),ncol(0),p(NULL),buf(NULL)
    {
        Create(nrow,ncol);
    }
    Array2dC(const Array2dC<T>& source);
    virtual ~Array2dC()
    {
        Clear();
    }

    Array2dC<T>& operator=(const Array2dC<T>& source);
    void Create(const int _nrow,const int _ncol);
    void Swap(Array2dC<T>& array2);
    void Zero(const T t = 0);
    void Clear();
};

template<class T>
Array2d<T>::Array2d(const Array2d<T>& source):nrow(0),ncol(0),p(NULL)
{
    if(source.p!=NULL)
    {
        Create(source.nrow,source.ncol);
        for(int i=0; i<nrow; i++) std::copy(source.p[i],source.p[i]+ncol,p[i]);
    }
}

template<class T>
Array2d<T>& Array2d<T>::operator=(const Array2d<T>& source)
{
    if(source.p!=NULL)
    {
        Create(source.nrow,source.ncol);
        for(int i=0; i<nrow; i++) std::copy(source.p[i],source.p[i]+ncol,p[i]);
    }
    else
        Clear();
    return *this;
}

template<class T>
void Array2d<T>::Create(const int _nrow,const int _ncol)
{
    assert(_nrow>0 && _ncol>0);
    Clear();
    nrow = _nrow;
    ncol = _ncol;
    p = new T*[nrow];
    assert(p!=NULL);
    for(int i=0; i<nrow; i++)
    {
        p[i] = new T[ncol];
        assert(p[i]!=NULL);
    }
}

template<class T>
void Array2d<T>::Swap(Array2d<T>& array2)
{
    std::swap(nrow,array2.nrow);
    std::swap(ncol,array2.ncol);
    std::swap(p,array2.p);
}

template<class T>
void Array2d<T>::Zero(const T t)
{
    if(nrow>0)
    {
        for(int i=0; i<nrow; i++) std::fill(p[i],p[i]+ncol,t);
    }
}

template<class T>
void Array2d<T>::Clear()
{
    for(int i=0; i<nrow; i++)
    {
        delete[] p[i];
        p[i] = NULL;
    }
    delete[] p;
    p = NULL;
    nrow = ncol = 0;
}

template<class T>
Array2dC<T>::Array2dC(const Array2dC<T>& source):nrow(0),ncol(0),p(NULL),buf(NULL)
{
    if(source.buf!=NULL)
    {
        Create(source.nrow,source.ncol);
        std::copy(source.buf,source.buf+nrow*ncol,buf);
    }
}

template<class T>
Array2dC<T>& Array2dC<T>::operator=(const Array2dC<T>& source)
{
    if(source.buf!=NULL)
    {
        Create(source.nrow,source.ncol);
        std::copy(source.buf,source.buf+nrow*ncol,buf);
    }
    else
        Clear();
    return *this;
}

template<class T>
void Array2dC<T>::Create(const int _nrow,const int _ncol)
{
    assert(_nrow>0 && _ncol>0);
    if(nrow==_nrow && ncol==_ncol) return;
    Clear();
    nrow = _nrow;
    ncol = _ncol;
    buf = new T[nrow*ncol];
    assert(buf!=NULL);
    p = new T*[nrow];
    assert(p!=NULL);
    for(int i=0; i<nrow; i++) p[i] = buf + i * ncol;
}

template<class T>
void Array2dC<T>::Swap(Array2dC<T>& array2)
{
    std::swap(nrow,array2.nrow);
    std::swap(ncol,array2.ncol);
    std::swap(p,array2.p);
    std::swap(buf,array2.buf);
}

template<class T>
void Array2dC<T>::Zero(const T t)
{
    if(nrow>0) std::fill(buf,buf+nrow*ncol,t);
}

template<class T>
void Array2dC<T>::Clear()
{
    delete[] buf;
    buf = NULL;
    delete[] p;
    p = NULL;
    nrow = ncol = 0;
}


/*****************************************/
// IntImage.h
/*****************************************/

template<class T>
class IntImage:public Array2dC<T>
{
private:
    IntImage(const IntImage<T> &source) { } // prohibit copy constructor

public:
    IntImage():variance(0.0),label(-1) { }
    virtual ~IntImage()
    {
        Clear();
    }

    virtual void Clear(void);
    inline void SetSize(const int h, const int w);
    bool Load(cv::Mat img);
    void Save(const std::string& filename) const;
    void Swap(IntImage<T>& image2);

    void CalcIntegralImageInPlace(void);
    void Resize(IntImage<T> &result,const REAL ratio) const;
    void Resize(IntImage<T>& result,const int height,const int width) const;

    IntImage<T>& operator=(const IntImage<T>& source);

    void Sobel(IntImage<REAL>& result,const bool useSqrt,const bool normalize);
public:
    using Array2dC<T>::nrow;
    using Array2dC<T>::ncol;
    using Array2dC<T>::buf;
    using Array2dC<T>::p;
    REAL variance;
    int label;
};

template<class T>
void IntImage<T>::Clear(void)
{
    Array2dC<T>::Clear();
    variance = 0.0;
    label = -1;
}

template<class T>
bool IntImage<T>::Load(cv::Mat img)
{
    if (img.empty()) return false;

    SetSize(img.rows, img.cols);
    for(int i=0,ih=img.rows,iw=img.cols; i<ih; i++)
    {
        T* pdata = p[i];
        unsigned char* pimg = reinterpret_cast<unsigned char*>(img.data+img.step*i);
        for(int j=0; j<iw; j++) pdata[j] = pimg[j];
    }

    return true;
}

template<class T>
void IntImage<T>::Save(const std::string& filename) const
{
    IplImage* img;

    img = cvCreateImage(cvSize(ncol,nrow),IPL_DEPTH_8U,1);
    for(int i=0,ih=img->height,iw=img->width; i<ih; i++)
    {
        T* pdata = p[i];
        unsigned char* pimg = reinterpret_cast<unsigned char*>(img->imageData+img->widthStep*i);
        for(int j=0; j<iw; j++) pimg[j] = (unsigned char)pdata[j];
    }
    cvSaveImage(filename.c_str(),img);
    cvReleaseImage(&img);
}

template<class T>
void IntImage<T>::SetSize(const int h,const int w)
{
    if((h == nrow) && (w == ncol)) return;
    Clear();
    Array2dC<T>::Create(h,w);
}

template<class T>
IntImage<T>& IntImage<T>::operator=(const IntImage<T>& source)
{
    if(&source==this) return *this;
    SetSize(source.nrow,source.ncol);
    std::copy(source.buf,source.buf+nrow*ncol,buf);
    label = source.label;
    variance = source.variance;
    return *this;
}

template<class T>
void IntImage<T>::Resize(IntImage<T> &result,const REAL ratio) const
{
    Resize(result,int(nrow*ratio),int(ncol*ratio));
}

template<class T>
void IntImage<T>::Resize(IntImage<T>& result,const int height,const int width) const
{
    assert(height>0 && width>0);
    result.SetSize(height,width);
    REAL ixratio = nrow*1.0/height, iyratio = ncol*1.0/width;

    REAL* p_y = new REAL[result.ncol];
    assert(p_y!=NULL);
    int* p_y0 = new int[result.ncol];
    assert(p_y0!=NULL);
    for(int i=0; i<width; i++)
    {
        p_y[i] = i*iyratio;
        p_y0[i] = (int)p_y[i];
        if(p_y0[i]==ncol-1) p_y0[i]--;
        p_y[i] -= p_y0[i];
    }

    for(int i=0; i<height; i++)
    {
        int x0;
        REAL x;
        x = i*ixratio;
        x0 = (int)x;
        if(x0==nrow-1) x0--;
        x -= x0;
        T* rp = result.p[i];
        const T* px0 = p[x0];
        const T* px1 = p[x0+1];
        for(int j=0; j<width; j++)
        {
            int y0=p_y0[j];
            REAL y=p_y[j],fx0,fx1;

            fx0 = REAL(px0[y0] + y*(px0[y0+1]-px0[y0]));
            fx1 = REAL(px1[y0] + y*(px1[y0+1]-px1[y0]));

            rp[j] = T(fx0 + x*(fx1-fx0));
        }
    }

    delete[] p_y;
    p_y=NULL;
    delete[] p_y0;
    p_y0=NULL;
}

template<class T>
void IntImage<T>::CalcIntegralImageInPlace(void)
// We pad a zero column and a zero row, so 24*24 image will be 25*25 in size
// if the input image is not padded, the results on 1st row will be problematic
{
    for(int i=1; i<ncol; i++)   // process the first line
    {
        buf[i] += buf[i-1];
    }
    for(int i=1; i<nrow; i++)
    {
        REAL partialsum = 0;
        T* curp = p[i];
        T* prep = p[i-1];
        for(int j=0; j<ncol; j++)
        {
            partialsum += REAL(curp[j]);
            curp[j] = prep[j] + partialsum;
        }
    }
}

template<class T>
void IntImage<T>::Swap(IntImage<T>& image2)
{
    Array2dC<T>::Swap(image2);
    std::swap(variance,image2.variance);
    std::swap(label,image2.label);
}

template<class T>
void IntImage<T>::Sobel(IntImage<REAL>& result,const bool useSqrt,const bool normalize)
{
    // compute the Sobel gradient. For now, we just use the very inefficient way. Optimization can be done later
// if useSqrt = true, we compute the real Sobel gradient; otherwise, the square of it
// if normalize = true, the numbers are normalized to be in 0..255
    result.Create(nrow,ncol);
    for(int i=0; i<nrow; i++) result.p[i][0] = result.p[i][ncol-1] = 0;
    std::fill(result.p[0],result.p[0]+ncol,0.0);
    std::fill(result.p[nrow-1],result.p[nrow-1],0.0);
    for(int i=1; i<nrow-1; i++)
    {
        T* p1 = p[i-1];
        T* p2 = p[i];
        T* p3 = p[i+1];
        REAL* pr = result.p[i];
        for(int j=1; j<ncol-1; j++)
        {
            REAL gx =     p1[j-1] - p1[j+1]
                          + 2*(p2[j-1]   - p2[j+1])
                          +    p3[j-1] - p3[j+1];
            REAL gy =     p1[j-1] - p3[j-1]
                          + 2*(p1[j]   - p3[j])
                          +    p1[j+1] - p3[j+1];
            pr[j] = gx*gx+gy*gy;
        }
    }
    if(useSqrt || normalize ) // if we want to normalize the result image, we'd better use the true Sobel gradient
        for(int i=1; i<nrow-1; i++)
            for(int j=1; j<ncol-1; j++)
                result.p[i][j] = sqrt(result.p[i][j]);

    if(normalize)
    {
        REAL minv = 1e20, maxv = -minv;
        for(int i=1; i<nrow-1; i++)
        {
            for(int j=1; j<ncol-1; j++)
            {
                if(result.p[i][j]<minv)
                    minv = result.p[i][j];
                else if(result.p[i][j]>maxv)
                    maxv = result.p[i][j];
            }
        }
        for(int i=0; i<nrow; i++) result.p[i][0] = result.p[i][ncol-1] = minv;
        for(int i=0; i<ncol; i++) result.p[0][i] = result.p[nrow-1][i] = minv;
        REAL s = 255.0/(maxv-minv);
        for(int i=0; i<nrow*ncol; i++) result.buf[i] = (result.buf[i]-minv)*s;
    }
}


/*****************************************/
// Pedestrian.h
/*****************************************/

class NodeDetector
{
public:
    enum NodeType { CD_LIN, CD_HIK, LINEAR, HISTOGRAM };
public:
    int type; // linear or histogram?
    Array2dC<double> classifier;
    double thresh;
    int featurelength;
    int upper_bound;
    int index;
    std::string filename;
public:
    NodeDetector(const NodeType _type,const int _featurelength,const int _upper_bound,const int _index,const char* _filename)
    {
        Load(_type,_featurelength,_upper_bound,_index,_filename);
        minvalue = DBL_MAX;
        maxvalue = -minvalue;
    }
    ~NodeDetector()
    {
    }

    void Load(const NodeType _type,const int _featurelength,const int _upper_bound,const int _index,const char* _filename);
    bool Classify(int* f);
private:
    double minvalue;
    double maxvalue;
public:
    void SetValues(const double v)
    {
        if(v>maxvalue) maxvalue = v;
        if(v<minvalue) minvalue = v;
    }
};

class CascadeDetector
{
public:
    int size;
    int length;
    NodeDetector** nodes;
public:

public:
    CascadeDetector()
        : size(20), length(0)
    {
        nodes = new NodeDetector*[size];
    }
    ~CascadeDetector()
    {
        for(int i=0; i<length; i++) delete nodes[i];
        delete[] nodes;
    }

    void AddNode(const NodeDetector::NodeType _type,const int _featurelength,const int _upper_bound,const char* _filename);
};

class DetectionScanner // who does the dirty jobs
{
public:
    DetectionScanner()
        : height(0), width(0), xdiv(0), ydiv(0), baseflength(0), ratio(0.0), cascade(NULL), integrals(NULL)
    {
    }
    DetectionScanner(const int _height,const int _width,const int _xdiv,const int _ydiv,
                     const int _baseflength,const double _ratio)
        :height(_height),width(_width),xdiv(_xdiv),ydiv(_ydiv),
         baseflength(_baseflength),ratio(_ratio),cascade(NULL),integrals(NULL)
    {
    }
    ~DetectionScanner()
    {
        delete cascade;
        delete[] integrals;
    }

    void LoadDetector(std::vector<NodeDetector::NodeType>& types,std::vector<int>& upper_bounds,std::vector<std::string>& filenames);

    int Scan(IntImage<double>& original,std::vector<cv::Rect>& results,const int stepsize,const int round,std::ofstream* out,const int upper_bound);
    int FastScan(IntImage<double>& original,std::vector<cv::Rect>& results,const int stepsize);
    int FeatureLength() const
    {
        return (xdiv-1)*(ydiv-1)*baseflength;
    }


    int height;
    int width;
    int xdiv;
    int ydiv;
    int baseflength;
    double ratio;
    static const int EXT = 1;

    CascadeDetector* cascade;

private:
    IntImage<double>* integrals;
    IntImage<double> image,sobel;
    IntImage<int> ct;
    Array2dC<int> hist;
    IntImage<double> scores;

    void InitImage(IntImage<double>& original);
    void InitIntegralImages(const int stepsize);
    void ResizeImage();
};

void RunFiles();

/*****************************************/
// Pedestrian_ICRA.cpp
/*****************************************/

// ---------------------------------------------------------------------
// Helper functions

// compute the Sobel image "ct" from "original"
void ComputeCT(IntImage<double>& original,IntImage<int>& ct)
{
    ct.Create(original.nrow,original.ncol);
    for(int i=2; i<original.nrow-2; i++)
    {
        double* p1 = original.p[i-1];
        double* p2 = original.p[i];
        double* p3 = original.p[i+1];
        int* ctp = ct.p[i];
        for(int j=2; j<original.ncol-2; j++)
        {
            int index = 0;
            if(p2[j]<=p1[j-1]) index += 0x80;
            if(p2[j]<=p1[j]) index += 0x40;
            if(p2[j]<=p1[j+1]) index += 0x20;
            if(p2[j]<=p2[j-1]) index += 0x10;
            if(p2[j]<=p2[j+1]) index += 0x08;
            if(p2[j]<=p3[j-1]) index += 0x04;
            if(p2[j]<=p3[j]) index += 0x02;
            if(p2[j]<=p3[j+1]) index ++;
            ctp[j] = index;
        }
    }
}

// Load SVM models -- linear SVM trained using LIBLINEAR
double UseSVM_CD_FastEvaluationStructure(const char* modelfile,const int m,Array2dC<double>& result)
{
    std::ifstream in(modelfile);
    if(in.good()==false)
    {
        std::cout<<"SVM model "<<modelfile<<" can not be loaded."<<std::endl;
        exit(-1);
    }
    std::string buffer;
    std::getline(in,buffer); // first line
    std::getline(in,buffer); // second line
    std::getline(in,buffer); // third line
    in>>buffer;
    assert(buffer=="nr_feature");
    int num_dim;
    in>>num_dim;
    assert(num_dim>0 && num_dim==m);
    std::getline(in,buffer); // end of line 4
    in>>buffer;
    assert(buffer=="bias");
    int bias;
    in>>bias;
    std::getline(in,buffer); //end of line 5;
    in>>buffer;
    assert(buffer=="w");
    std::getline(in,buffer); //end of line 6
    result.Create(1,num_dim);
    for(int i=0; i<num_dim; i++) in>>result.buf[i];
    double rho = 0;
    if(bias>=0) in>>rho;
    in.close();
    return rho;
}

// Load SVM models -- Histogram Intersectin Kernel SVM trained by libHIK
double UseSVM_CD_FastEvaluationStructure(const char* modelfile, const int m, const int upper_bound, Array2dC<double>& result)
{

    std::ifstream fs(modelfile, std::fstream::binary);
	if( !fs.is_open() )
	{
		std::cout << "SVM model " << modelfile << " can not be loaded." << std::endl;
		exit(-1);
	}
    // Header
    int rows, cols, type, channels;
    fs.read((char*)&rows, sizeof(int));         // rows
    fs.read((char*)&cols, sizeof(int));         // cols
    fs.read((char*)&type, sizeof(int));         // type
    fs.read((char*)&channels, sizeof(int));     // channels

    // Data
    cv::Mat mat(rows, cols, type);
    fs.read((char*)mat.data, CV_ELEM_SIZE(type) * rows * cols);

    int num_dim = m;

    result.Create(num_dim, upper_bound);
    for(int i=0; i<num_dim; i++)
        for (int j = 0; j < upper_bound; j++)
        {
            result.p[i][j]= mat.at<double>(i, j);
        }

    return -0.00455891;
}

// End of Helper functions
// ---------------------------------------------------------------------

// ---------------------------------------------------------------------
// Functions that load the two classifiers
inline void LoadCascade(std::string cascade1, std::string cascade2, DetectionScanner& ds)
{
    std::vector<NodeDetector::NodeType> types;
    std::vector<int> upper_bounds;
    std::vector<std::string> filenames;

    types.push_back(NodeDetector::CD_LIN); // first node
    upper_bounds.push_back(100);
    filenames.push_back(cascade1);
    types.push_back(NodeDetector::CD_HIK); // second node
    upper_bounds.push_back(353);
    filenames.push_back(cascade2);

    ds.LoadDetector(types,upper_bounds,filenames);
    // You can adjust these parameters for different speed, accuracy etc
    ds.cascade->nodes[0]->thresh += 0.8;
    ds.cascade->nodes[1]->thresh -= 0.095;
}

void DetectionScanner::LoadDetector(std::vector<NodeDetector::NodeType>& types,std::vector<int>& upper_bounds,std::vector<std::string>& filenames)
{
    unsigned int depth = types.size();
    assert(depth>0 && depth==upper_bounds.size() && depth==filenames.size());
    if(cascade)
        delete cascade;
    cascade = new CascadeDetector;
    assert(xdiv>0 && ydiv>0);
    for(unsigned int i=0; i<depth; i++)
        cascade->AddNode(types[i],(xdiv-EXT)*(ydiv-EXT)*baseflength,upper_bounds[i],filenames[i].c_str());

    hist.Create(1,baseflength*(xdiv-EXT)*(ydiv-EXT));
}

void NodeDetector::Load(const NodeType _type,const int _featurelength,const int _upper_bound,const int _index,const char* _filename)
{
    type = _type;
    index = _index;
    filename = _filename;
    featurelength = _featurelength;
    upper_bound = _upper_bound;
    if(type==CD_LIN)
        thresh = UseSVM_CD_FastEvaluationStructure(_filename,_featurelength,classifier);
    else if(type==CD_HIK)
        thresh = UseSVM_CD_FastEvaluationStructure(_filename,_featurelength,upper_bound,classifier);

    if(type==CD_LIN) type = LINEAR;
    if(type==CD_HIK) type = HISTOGRAM;
}

void CascadeDetector::AddNode(const NodeDetector::NodeType _type,const int _featurelength,const int _upper_bound,const char* _filename)
{
    if(length==size)
    {
        int newsize = size * 2;
        NodeDetector** p = new NodeDetector*[newsize];
        assert(p!=NULL);
        std::copy(nodes,nodes+size,p);
        size = newsize;
        delete[] nodes;
        nodes = p;
    }
    nodes[length] = new NodeDetector(_type,_featurelength,_upper_bound,length,_filename);
    length++;
}

// End of functions that load the two classifiers
// ---------------------------------------------------------------------

// ---------------------------------------------------------------------
// Detection functions

// initialization -- compute the Census Tranform image for CENTRIST
void DetectionScanner::InitImage(IntImage<double>& original)
{
    image = original;
    image.Sobel(sobel,false,false);
    ComputeCT(sobel,ct);
}

// combine the (xdiv-1)*(ydiv-1) integral images into a single one
void DetectionScanner::InitIntegralImages(const int stepsize)
{
    if(cascade->nodes[0]->type!=NodeDetector::LINEAR)
        return; // No need to prepare integral images

    const int hd = height/xdiv*2-2;
    const int wd = width/ydiv*2-2;
    scores.Create(ct.nrow,ct.ncol);
    scores.Zero(cascade->nodes[0]->thresh/hd/wd);
    double* linearweights = cascade->nodes[0]->classifier.buf;
    for(int i=0; i<xdiv-EXT; i++)
    {
        const int xoffset = height/xdiv*i;
        for(int j=0; j<ydiv-EXT; j++)
        {
            const int yoffset = width/ydiv*j;
            for(int x=2; x<ct.nrow-2-xoffset; x++)
            {
                int* ctp = ct.p[x+xoffset]+yoffset;
                double* tempp = scores.p[x];
                for(int y=2; y<ct.ncol-2-yoffset; y++)
                    tempp[y] += linearweights[ctp[y]];
            }
            linearweights += baseflength;
        }
    }
    scores.CalcIntegralImageInPlace();
    for(int i=2; i<ct.nrow-2-height; i+=stepsize)
    {
        double* p1 = scores.p[i];
        double* p2 = scores.p[i+hd];
        for(int j=2; j<ct.ncol-2-width; j+=stepsize)
            p1[j] += (p2[j+wd] - p2[j] - p1[j+wd]);
    }
}

// Resize the input image and then re-compute Sobel image etc
void DetectionScanner::ResizeImage()
{
    image.Resize(sobel,ratio);
    image.Swap(sobel);
    image.Sobel(sobel,false,false);
    ComputeCT(sobel,ct);
}

// The function that does the real detection
int DetectionScanner::FastScan(IntImage<double>& original,std::vector<cv::Rect>& results,const int stepsize)
{
    if(original.nrow<height+5 || original.ncol<width+5) return 0;
    const int hd = height/xdiv;
    const int wd = width/ydiv;
    InitImage(original);
    results.clear();

    hist.Create(1,baseflength*(xdiv-EXT)*(ydiv-EXT));

    NodeDetector* node = cascade->nodes[1];
    double** pc = node->classifier.p;
    int oheight = original.nrow, owidth = original.ncol;
    cv::Rect rect;
    while(image.nrow>=height && image.ncol>=width)
    {
        InitIntegralImages(stepsize);
        for(int i=2; i+height<image.nrow-2; i+=stepsize)
        {
            const double* sp = scores.p[i];
            for(int j=2; j+width<image.ncol-2; j+=stepsize)
            {
                if(sp[j]<=0) continue;
                int* p = hist.buf;
                hist.Zero();
                for(int k=0; k<xdiv-EXT; k++)
                {
                    for(int t=0; t<ydiv-EXT; t++)
                    {
                        for(int x=i+k*hd+1; x<i+(k+1+EXT)*hd-1; x++)
                        {
                            int* ctp = ct.p[x];
                            for(int y=j+t*wd+1; y<j+(t+1+EXT)*wd-1; y++)
                                p[ctp[y]]++;
                        }
                        p += baseflength;
                    }
                }
                double score = node->thresh;
                for(int k=0; k<node->classifier.nrow; k++) score += pc[k][hist.buf[k]];
                if(score>0)
                {
                    rect.y = i * oheight / image.nrow;
                    rect.height = (oheight * height) / image.nrow + 1;
                    rect.x = j * owidth / image.ncol;
                    rect.width = (width * owidth) /image.ncol + 1;
                    results.push_back(rect);
                }
            }
        }
        ResizeImage();
    }
    return 0;
}

// End of Detection functions
// ---------------------------------------------------------------------
