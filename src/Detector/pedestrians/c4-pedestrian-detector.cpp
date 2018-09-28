#include "c4-pedestrian-detector.h"

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
double UseSVM_CD_FastEvaluationStructure(const char* modelfile, const int m, Array2dC<double>& result)
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
    int num_dim = m;
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
void LoadCascade(std::string cascade1, std::string cascade2, DetectionScanner& ds)
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
