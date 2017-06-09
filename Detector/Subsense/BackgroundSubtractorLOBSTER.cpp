#include "BackgroundSubtractorLOBSTER.h"
#include "DistanceUtils.h"
#include "RandUtils.h"
#include <iostream>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iomanip>

BackgroundSubtractorLOBSTER::BackgroundSubtractorLOBSTER(	 float fRelLBSPThreshold
															,size_t nLBSPThresholdOffset
															,size_t nDescDistThreshold
															,size_t nColorDistThreshold
															,size_t nBGSamples
															,size_t nRequiredBGSamples)
	:	 BackgroundSubtractorLBSP(fRelLBSPThreshold,nLBSPThresholdOffset)
		,m_nColorDistThreshold(nColorDistThreshold)
		,m_nDescDistThreshold(nDescDistThreshold)
		,m_nBGSamples(nBGSamples)
		,m_nRequiredBGSamples(nRequiredBGSamples) {
	CV_Assert(m_nRequiredBGSamples<=m_nBGSamples);
	m_bAutoModelResetEnabled = false; // @@@@@@ not supported here for now
}

BackgroundSubtractorLOBSTER::~BackgroundSubtractorLOBSTER() {
	if(m_aPxIdxLUT)
		delete[] m_aPxIdxLUT;
	if(m_aPxInfoLUT)
	    delete[] m_aPxInfoLUT;
}

void BackgroundSubtractorLOBSTER::initialize(const cv::Mat& oInitImg, const cv::Mat& oROI) {
	CV_Assert(!oInitImg.empty() && oInitImg.cols>0 && oInitImg.rows>0);
	CV_Assert(oInitImg.isContinuous());
	CV_Assert(oInitImg.type()==CV_8UC1 || oInitImg.type()==CV_8UC3);
	if(oInitImg.type()==CV_8UC3) {
		std::vector<cv::Mat> voInitImgChannels;
		cv::split(oInitImg,voInitImgChannels);
		if(!cv::countNonZero((voInitImgChannels[0]!=voInitImgChannels[1])|(voInitImgChannels[2]!=voInitImgChannels[1])))
			std::cout << std::endl << "\tBackgroundSubtractorLOBSTER : Warning, grayscale images should always be passed in CV_8UC1 format for optimal performance." << std::endl;
	}
	cv::Mat oNewBGROI;
	if(oROI.empty() && (m_oROI.empty() || oROI.size()!=oInitImg.size())) {
		oNewBGROI.create(oInitImg.size(),CV_8UC1);
		oNewBGROI = cv::Scalar_<uchar>(UCHAR_MAX);
	}
	else if(oROI.empty())
		oNewBGROI = m_oROI;
	else {
		CV_Assert(oROI.size()==oInitImg.size() && oROI.type()==CV_8UC1);
		CV_Assert(cv::countNonZero((oROI<UCHAR_MAX)&(oROI>0))==0);
		oNewBGROI = oROI.clone();
	}
	LBSP::validateROI(oNewBGROI);
	const size_t nROIPxCount = (size_t)cv::countNonZero(oNewBGROI);
	CV_Assert(nROIPxCount>0);
	m_oROI = oNewBGROI;
	m_oImgSize = oInitImg.size();
	m_nImgType = oInitImg.type();
	m_nImgChannels = oInitImg.channels();
	m_nTotPxCount = m_oImgSize.area();
	m_nTotRelevantPxCount = nROIPxCount;
	m_nFrameIndex = 0;
	m_nFramesSinceLastReset = 0;
	m_nModelResetCooldown = 0;
	m_oLastFGMask.create(m_oImgSize,CV_8UC1);
	m_oLastFGMask = cv::Scalar_<uchar>(0);
	m_oLastColorFrame.create(m_oImgSize,CV_8UC((int)m_nImgChannels));
	m_oLastColorFrame = cv::Scalar_<uchar>::all(0);
	m_oLastDescFrame.create(m_oImgSize,CV_16UC((int)m_nImgChannels));
	m_oLastDescFrame = cv::Scalar_<ushort>::all(0);
	m_voBGColorSamples.resize(m_nBGSamples);
	m_voBGDescSamples.resize(m_nBGSamples);
	for(size_t s=0; s<m_nBGSamples; ++s) {
		m_voBGColorSamples[s].create(m_oImgSize,CV_8UC((int)m_nImgChannels));
		m_voBGColorSamples[s] = cv::Scalar_<uchar>::all(0);
		m_voBGDescSamples[s].create(m_oImgSize,CV_16UC((int)m_nImgChannels));
		m_voBGDescSamples[s] = cv::Scalar_<ushort>::all(0);
	}
	if(m_aPxIdxLUT)
		delete[] m_aPxIdxLUT;
	if(m_aPxInfoLUT)
	    delete[] m_aPxInfoLUT;
	m_aPxIdxLUT = new size_t[m_nTotRelevantPxCount];
	m_aPxInfoLUT = new PxInfoBase[m_nTotPxCount];
	if(m_nImgChannels==1) {
		CV_Assert(m_oLastColorFrame.step.p[0]==(size_t)m_oImgSize.width && m_oLastColorFrame.step.p[1]==1);
		CV_Assert(m_oLastDescFrame.step.p[0]==m_oLastColorFrame.step.p[0]*2 && m_oLastDescFrame.step.p[1]==m_oLastColorFrame.step.p[1]*2);
		for(size_t t=0; t<=UCHAR_MAX; ++t)
			m_anLBSPThreshold_8bitLUT[t] = cv::saturate_cast<uchar>((t*m_fRelLBSPThreshold+m_nLBSPThresholdOffset)/2);
		for(size_t nPxIter=0, nModelIter=0; nPxIter<m_nTotPxCount; ++nPxIter) {
			if(m_oROI.data[nPxIter]) {
				m_aPxIdxLUT[nModelIter] = nPxIter;
				m_aPxInfoLUT[nPxIter].nImgCoord_Y = (int)nPxIter/m_oImgSize.width;
				m_aPxInfoLUT[nPxIter].nImgCoord_X = (int)nPxIter%m_oImgSize.width;
				m_aPxInfoLUT[nPxIter].nModelIdx = nModelIter;
				m_oLastColorFrame.data[nPxIter] = oInitImg.data[nPxIter];
				const size_t nDescIter = nPxIter*2;
				LBSP::computeGrayscaleDescriptor(oInitImg,oInitImg.data[nPxIter],m_aPxInfoLUT[nPxIter].nImgCoord_X,m_aPxInfoLUT[nPxIter].nImgCoord_Y,m_anLBSPThreshold_8bitLUT[oInitImg.data[nPxIter]],*((ushort*)(m_oLastDescFrame.data+nDescIter)));
				++nModelIter;
			}
		}
	}
	else { //m_nImgChannels==3
		CV_Assert(m_oLastColorFrame.step.p[0]==(size_t)m_oImgSize.width*3 && m_oLastColorFrame.step.p[1]==3);
		CV_Assert(m_oLastDescFrame.step.p[0]==m_oLastColorFrame.step.p[0]*2 && m_oLastDescFrame.step.p[1]==m_oLastColorFrame.step.p[1]*2);
		for(size_t t=0; t<=UCHAR_MAX; ++t)
			m_anLBSPThreshold_8bitLUT[t] = cv::saturate_cast<uchar>(t*m_fRelLBSPThreshold+m_nLBSPThresholdOffset);
		for(size_t nPxIter=0, nModelIter=0; nPxIter<m_nTotPxCount; ++nPxIter) {
			if(m_oROI.data[nPxIter]) {
				m_aPxIdxLUT[nModelIter] = nPxIter;
				m_aPxInfoLUT[nPxIter].nImgCoord_Y = (int)nPxIter/m_oImgSize.width;
				m_aPxInfoLUT[nPxIter].nImgCoord_X = (int)nPxIter%m_oImgSize.width;
				m_aPxInfoLUT[nPxIter].nModelIdx = nModelIter;
				const size_t nPxRGBIter = nPxIter*3;
				const size_t nDescRGBIter = nPxRGBIter*2;
				for(size_t c=0; c<3; ++c) {
					m_oLastColorFrame.data[nPxRGBIter+c] = oInitImg.data[nPxRGBIter+c];
					LBSP::computeSingleRGBDescriptor(oInitImg,oInitImg.data[nPxRGBIter+c],m_aPxInfoLUT[nPxIter].nImgCoord_X,m_aPxInfoLUT[nPxIter].nImgCoord_Y,c,m_anLBSPThreshold_8bitLUT[oInitImg.data[nPxRGBIter+c]],((ushort*)(m_oLastDescFrame.data+nDescRGBIter))[c]);
				}
				++nModelIter;
			}
		}
	}
	m_bInitialized = true;
	refreshModel(1.0f);
}

void BackgroundSubtractorLOBSTER::refreshModel(float fSamplesRefreshFrac, bool bForceFGUpdate) {
	// == refresh
	CV_Assert(m_bInitialized);
	CV_Assert(fSamplesRefreshFrac>0.0f && fSamplesRefreshFrac<=1.0f);
	const size_t nModelsToRefresh = fSamplesRefreshFrac<1.0f?(size_t)(fSamplesRefreshFrac*m_nBGSamples):m_nBGSamples;
	const size_t nRefreshStartPos = fSamplesRefreshFrac<1.0f?rand()%m_nBGSamples:0;
	if(m_nImgChannels==1) {
		for(size_t nModelIter=0; nModelIter<m_nTotRelevantPxCount; ++nModelIter) {
			const size_t nPxIter = m_aPxIdxLUT[nModelIter];
			if(bForceFGUpdate || !m_oLastFGMask.data[nPxIter]) {
				for(size_t nCurrModelIdx=nRefreshStartPos; nCurrModelIdx<nRefreshStartPos+nModelsToRefresh; ++nCurrModelIdx) {
					int nSampleImgCoord_Y, nSampleImgCoord_X;
					getRandSamplePosition(nSampleImgCoord_X,nSampleImgCoord_Y,m_aPxInfoLUT[nPxIter].nImgCoord_X,m_aPxInfoLUT[nPxIter].nImgCoord_Y,LBSP::PATCH_SIZE/2,m_oImgSize);
					const size_t nSamplePxIdx = m_oImgSize.width*nSampleImgCoord_Y + nSampleImgCoord_X;
					if(bForceFGUpdate || !m_oLastFGMask.data[nSamplePxIdx]) {
						const size_t nCurrRealModelIdx = nCurrModelIdx%m_nBGSamples;
						m_voBGColorSamples[nCurrRealModelIdx].data[nPxIter] = m_oLastColorFrame.data[nSamplePxIdx];
						*((ushort*)(m_voBGDescSamples[nCurrRealModelIdx].data+nPxIter*2)) = *((ushort*)(m_oLastDescFrame.data+nSamplePxIdx*2));
					}
				}
			}
		}
	}
	else { //m_nImgChannels==3
		for(size_t nModelIter=0; nModelIter<m_nTotRelevantPxCount; ++nModelIter) {
			const size_t nPxIter = m_aPxIdxLUT[nModelIter];
			if(bForceFGUpdate || !m_oLastFGMask.data[nPxIter]) {
				for(size_t nCurrModelIdx=nRefreshStartPos; nCurrModelIdx<nRefreshStartPos+nModelsToRefresh; ++nCurrModelIdx) {
					int nSampleImgCoord_Y, nSampleImgCoord_X;
					getRandSamplePosition(nSampleImgCoord_X,nSampleImgCoord_Y,m_aPxInfoLUT[nPxIter].nImgCoord_X,m_aPxInfoLUT[nPxIter].nImgCoord_Y,LBSP::PATCH_SIZE/2,m_oImgSize);
					const size_t nSamplePxIdx = m_oImgSize.width*nSampleImgCoord_Y + nSampleImgCoord_X;
					if(bForceFGUpdate || !m_oLastFGMask.data[nSamplePxIdx]) {
						const size_t nCurrRealModelIdx = nCurrModelIdx%m_nBGSamples;
						for(size_t c=0; c<3; ++c) {
							m_voBGColorSamples[nCurrRealModelIdx].data[nPxIter*3+c] = m_oLastColorFrame.data[nSamplePxIdx*3+c];
							*((ushort*)(m_voBGDescSamples[nCurrRealModelIdx].data+(nPxIter*3+c)*2)) = *((ushort*)(m_oLastDescFrame.data+(nSamplePxIdx*3+c)*2));
						}
					}
				}
			}
		}
	}
}

void BackgroundSubtractorLOBSTER::operator()(cv::InputArray _image, cv::OutputArray _fgmask, double learningRate) {
	CV_Assert(m_bInitialized);
	CV_Assert(learningRate>0);
	cv::Mat oInputImg = _image.getMat();
	CV_Assert(oInputImg.type()==m_nImgType && oInputImg.size()==m_oImgSize);
	CV_Assert(oInputImg.isContinuous());
	_fgmask.create(m_oImgSize,CV_8UC1);
	cv::Mat oCurrFGMask = _fgmask.getMat();
	oCurrFGMask = cv::Scalar_<uchar>(0);
	const size_t nLearningRate = (size_t)ceil(learningRate);
	if(m_nImgChannels==1) {
		for(size_t nModelIter=0; nModelIter<m_nTotRelevantPxCount; ++nModelIter) {
			const size_t nPxIter = m_aPxIdxLUT[nModelIter];
			const size_t nDescIter = nPxIter*2;
			const int nCurrImgCoord_X = m_aPxInfoLUT[nPxIter].nImgCoord_X;
			const int nCurrImgCoord_Y = m_aPxInfoLUT[nPxIter].nImgCoord_Y;
			const uchar nCurrColor = oInputImg.data[nPxIter];
			size_t nGoodSamplesCount=0, nModelIdx=0;
			ushort nCurrInputDesc;
			while(nGoodSamplesCount<m_nRequiredBGSamples && nModelIdx<m_nBGSamples) {
				const uchar nBGColor = m_voBGColorSamples[nModelIdx].data[nPxIter];
				{
					const size_t nColorDist = L1dist(nCurrColor,nBGColor);
					if(nColorDist>m_nColorDistThreshold/2)
						goto failedcheck1ch;
					LBSP::computeGrayscaleDescriptor(oInputImg,nBGColor,nCurrImgCoord_X,nCurrImgCoord_Y,m_anLBSPThreshold_8bitLUT[nBGColor],nCurrInputDesc);
					const size_t nDescDist = hdist(nCurrInputDesc,*((ushort*)(m_voBGDescSamples[nModelIdx].data+nDescIter)));
					if(nDescDist>m_nDescDistThreshold)
						goto failedcheck1ch;
					nGoodSamplesCount++;
				}
				failedcheck1ch:
				nModelIdx++;
			}
			if(nGoodSamplesCount<m_nRequiredBGSamples)
				oCurrFGMask.data[nPxIter] = UCHAR_MAX;
			else {
				if((rand()%nLearningRate)==0) {
					const size_t nSampleModelIdx = rand()%m_nBGSamples;
					ushort& nRandInputDesc = *((ushort*)(m_voBGDescSamples[nSampleModelIdx].data+nDescIter));
					LBSP::computeGrayscaleDescriptor(oInputImg,nCurrColor,nCurrImgCoord_X,nCurrImgCoord_Y,m_anLBSPThreshold_8bitLUT[nCurrColor],nRandInputDesc);
					m_voBGColorSamples[nSampleModelIdx].data[nPxIter] = nCurrColor;
				}
				if((rand()%nLearningRate)==0) {
					int nSampleImgCoord_Y, nSampleImgCoord_X;
					getRandNeighborPosition_3x3(nSampleImgCoord_X,nSampleImgCoord_Y,nCurrImgCoord_X,nCurrImgCoord_Y,LBSP::PATCH_SIZE/2,m_oImgSize);
					const size_t nSampleModelIdx = rand()%m_nBGSamples;
					ushort& nRandInputDesc = m_voBGDescSamples[nSampleModelIdx].at<ushort>(nSampleImgCoord_Y,nSampleImgCoord_X);
					LBSP::computeGrayscaleDescriptor(oInputImg,nCurrColor,nCurrImgCoord_X,nCurrImgCoord_Y,m_anLBSPThreshold_8bitLUT[nCurrColor],nRandInputDesc);
					m_voBGColorSamples[nSampleModelIdx].at<uchar>(nSampleImgCoord_Y,nSampleImgCoord_X) = nCurrColor;
				}
			}
		}
	}
	else { //m_nImgChannels==3
		const size_t nCurrDescDistThreshold = m_nDescDistThreshold*3;
		const size_t nCurrColorDistThreshold = m_nColorDistThreshold*3;
		const size_t nCurrSCDescDistThreshold = nCurrDescDistThreshold/2;
		const size_t nCurrSCColorDistThreshold = nCurrColorDistThreshold/2;
		const size_t desc_row_step = m_voBGDescSamples[0].step.p[0];
		const size_t img_row_step = m_voBGColorSamples[0].step.p[0];
		for(size_t nModelIter=0; nModelIter<m_nTotRelevantPxCount; ++nModelIter) {
			const size_t nPxIter = m_aPxIdxLUT[nModelIter];
			const int nCurrImgCoord_X = m_aPxInfoLUT[nPxIter].nImgCoord_X;
			const int nCurrImgCoord_Y = m_aPxInfoLUT[nPxIter].nImgCoord_Y;
			const size_t nPxIterRGB = nPxIter*3;
			const size_t nDescIterRGB = nPxIterRGB*2;
			const uchar* const anCurrColor = oInputImg.data+nPxIterRGB;
			size_t nGoodSamplesCount=0, nModelIdx=0;
			ushort anCurrInputDesc[3];
			while(nGoodSamplesCount<m_nRequiredBGSamples && nModelIdx<m_nBGSamples) {
				const ushort* const anBGDesc = (ushort*)(m_voBGDescSamples[nModelIdx].data+nDescIterRGB);
				const uchar* const anBGColor = m_voBGColorSamples[nModelIdx].data+nPxIterRGB;
				size_t nTotColorDist = 0;
				size_t nTotDescDist = 0;
				for(size_t c=0;c<3; ++c) {
					const size_t nColorDist = L1dist(anCurrColor[c],anBGColor[c]);
					if(nColorDist>nCurrSCColorDistThreshold)
						goto failedcheck3ch;
					LBSP::computeSingleRGBDescriptor(oInputImg,anBGColor[c],nCurrImgCoord_X,nCurrImgCoord_Y,c,m_anLBSPThreshold_8bitLUT[anBGColor[c]],anCurrInputDesc[c]);
					const size_t nDescDist = hdist(anCurrInputDesc[c],anBGDesc[c]);
					if(nDescDist>nCurrSCDescDistThreshold)
						goto failedcheck3ch;
					nTotColorDist += nColorDist;
					nTotDescDist += nDescDist;
				}
				if(nTotDescDist<=nCurrDescDistThreshold && nTotColorDist<=nCurrColorDistThreshold)
					nGoodSamplesCount++;
				failedcheck3ch:
				nModelIdx++;
			}
			if(nGoodSamplesCount<m_nRequiredBGSamples)
				oCurrFGMask.data[nPxIter] = UCHAR_MAX;
			else {
				if((rand()%nLearningRate)==0) {
					const size_t nSampleModelIdx = rand()%m_nBGSamples;
					ushort* anRandInputDesc = ((ushort*)(m_voBGDescSamples[nSampleModelIdx].data+nDescIterRGB));
					const size_t anCurrIntraLBSPThresholds[3] = {m_anLBSPThreshold_8bitLUT[anCurrColor[0]],m_anLBSPThreshold_8bitLUT[anCurrColor[1]],m_anLBSPThreshold_8bitLUT[anCurrColor[2]]};
					LBSP::computeRGBDescriptor(oInputImg,anCurrColor,nCurrImgCoord_X,nCurrImgCoord_Y,anCurrIntraLBSPThresholds,anRandInputDesc);
					for(size_t c=0; c<3; ++c)
						*(m_voBGColorSamples[nSampleModelIdx].data+nPxIterRGB+c) = anCurrColor[c];
				}
				if((rand()%nLearningRate)==0) {
					int nSampleImgCoord_Y, nSampleImgCoord_X;
					getRandNeighborPosition_3x3(nSampleImgCoord_X,nSampleImgCoord_Y,nCurrImgCoord_X,nCurrImgCoord_Y,LBSP::PATCH_SIZE/2,m_oImgSize);
					const size_t nSampleModelIdx = rand()%m_nBGSamples;
					ushort* anRandInputDesc = ((ushort*)(m_voBGDescSamples[nSampleModelIdx].data + desc_row_step*nSampleImgCoord_Y + 6*nSampleImgCoord_X));
					const size_t anCurrIntraLBSPThresholds[3] = {m_anLBSPThreshold_8bitLUT[anCurrColor[0]],m_anLBSPThreshold_8bitLUT[anCurrColor[1]],m_anLBSPThreshold_8bitLUT[anCurrColor[2]]};
					LBSP::computeRGBDescriptor(oInputImg,anCurrColor,nCurrImgCoord_X,nCurrImgCoord_Y,anCurrIntraLBSPThresholds,anRandInputDesc);
					for(size_t c=0; c<3; ++c)
						*(m_voBGColorSamples[nSampleModelIdx].data + img_row_step*nSampleImgCoord_Y + 3*nSampleImgCoord_X + c) = anCurrColor[c];
				}
			}
		}
	}
	cv::medianBlur(oCurrFGMask,m_oLastFGMask,m_nDefaultMedianBlurKernelSize);
	m_oLastFGMask.copyTo(oCurrFGMask);
}

void BackgroundSubtractorLOBSTER::getBackgroundImage(cv::OutputArray backgroundImage) const {
	CV_DbgAssert(m_bInitialized);
	cv::Mat oAvgBGImg = cv::Mat::zeros(m_oImgSize,CV_32FC((int)m_nImgChannels));
	for(size_t s=0; s<m_nBGSamples; ++s) {
		for(int y=0; y<m_oImgSize.height; ++y) {
			for(int x=0; x<m_oImgSize.width; ++x) {
				const size_t idx_nimg = m_voBGColorSamples[s].step.p[0]*y + m_voBGColorSamples[s].step.p[1]*x;
				const size_t idx_flt32 = idx_nimg*4;
				float* oAvgBgImgPtr = (float*)(oAvgBGImg.data+idx_flt32);
				const uchar* const oBGImgPtr = m_voBGColorSamples[s].data+idx_nimg;
				for(size_t c=0; c<m_nImgChannels; ++c)
					oAvgBgImgPtr[c] += ((float)oBGImgPtr[c])/m_nBGSamples;
			}
		}
	}
	oAvgBGImg.convertTo(backgroundImage,CV_8U);
}

void BackgroundSubtractorLOBSTER::getBackgroundDescriptorsImage(cv::OutputArray backgroundDescImage) const {
	CV_Assert(LBSP::DESC_SIZE==2);
	CV_Assert(m_bInitialized);
	cv::Mat oAvgBGDesc = cv::Mat::zeros(m_oImgSize,CV_32FC((int)m_nImgChannels));
	for(size_t n=0; n<m_voBGDescSamples.size(); ++n) {
		for(int y=0; y<m_oImgSize.height; ++y) {
			for(int x=0; x<m_oImgSize.width; ++x) {
				const size_t idx_ndesc = m_voBGDescSamples[n].step.p[0]*y + m_voBGDescSamples[n].step.p[1]*x;
				const size_t idx_flt32 = idx_ndesc*2;
				float* oAvgBgDescPtr = (float*)(oAvgBGDesc.data+idx_flt32);
				const ushort* const oBGDescPtr = (ushort*)(m_voBGDescSamples[n].data+idx_ndesc);
				for(size_t c=0; c<m_nImgChannels; ++c)
					oAvgBgDescPtr[c] += ((float)oBGDescPtr[c])/m_voBGDescSamples.size();
			}
		}
	}
	oAvgBGDesc.convertTo(backgroundDescImage,CV_16U);
}

void BackgroundSubtractorLOBSTER::apply(cv::InputArray image, cv::OutputArray fgmask, double learningRateOverride)
{
    if(learningRateOverride<0)
    {
        learningRateOverride = BGSLOBSTER_DEFAULT_LEARNING_RATE;
    }
    (*this)(image, fgmask, learningRateOverride);
}
