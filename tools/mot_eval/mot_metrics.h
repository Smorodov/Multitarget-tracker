#pragma once

#include <map>
#include <set>
#include <string>
#include <vector>
#include <opencv2/core.hpp>

///
/// \brief One MOT row: GT or tracker result.
///
/// GT files (MOTChallenge, VisDrone-MOT) are
///     frame,id,x,y,w,h,conf,class,...      (1-based frames)
/// result files are
///     frame,id,x,y,w,h,conf,-1,-1,-1
///
struct MotDetection
{
    int frame = 0;      // frame number as written in the file
    int id = 0;         // track id
    cv::Rect box;       // x, y, w, h (top-left corner)
    float conf = -1.f;
    int cls = -1;       // class id; -1 when the row has no class (results)
};

///
/// \brief Aggregated metrics of one sequence.
///
/// Semantics follow the py-motmetrics / MOTChallenge devkit convention:
/// CLEAR MOT with continuity-prior matching and ID measures on a global
/// min-cost assignment (Ristani et al. 2016).
///
struct MotMetrics
{
    long long numFrames = 0;
    long long numObjects = 0;        // evaluated GT boxes
    long long numPredictions = 0;    // result boxes (after filtering/clamping)
    long long numMatches = 0;
    long long numMisses = 0;         // FN
    long long numFalsePositives = 0; // FP (not suppressed by ignore regions)
    long long numSuppressed = 0;     // FPs removed by ignore GT regions
    long long numSwitches = 0;       // IDSW
    double sumMatchDist = 0.0;       // sum of (1 - IoU) over matches

    long long idtp = 0;
    long long idfp = 0;
    long long idfn = 0;

    ///
    double Mota() const;      // 1 - (FN + FP + IDSW) / numObjects; may be negative
    double Motp() const;      // py-motmetrics convention: avg (1 - IoU), lower is better
    double MotpOverlap() const; // avg IoU of matches (leaderboard MOTP), higher is better
    double Idf1() const;      // 2 * IDTP / (numObjects + numPredictions)
};

///
struct MotEvalOptions
{
    float minIou = 0.5f;
    std::set<int> evalClasses;   // GT classes to evaluate; empty = all classes
    std::set<int> ignoreClasses; // GT classes whose boxes suppress FPs (and are never missed)
    int imageWidth = 0;          // clamp boxes to the frame when > 0
    int imageHeight = 0;
    long long minResArea = 0;    // drop result boxes smaller than this (px^2);
                                 // standard practice for VisDrone-style scenes
                                 // where the detector finds many objects that
                                 // the GT does not annotate
};

///
/// \brief Evaluate one sequence.
///
/// GT rows of classes that are neither evaluated nor ignored are dropped
/// entirely (not counted as GT, no FP suppression). Results rows are not
/// class-filtered (the detector whitelist is the place for that).
///
/// A result box is suppressed (not counted as FP) inside an ignore-class GT
/// box when it overlaps with IoU >= minIou OR when its center lies inside
/// the ignore box (ignore regions are often large areas where the plain
/// IoU test is meaningless).
///
MotMetrics EvaluateMotSequence(const std::vector<MotDetection>& gt,
                               const std::vector<MotDetection>& res,
                               const MotEvalOptions& options);
