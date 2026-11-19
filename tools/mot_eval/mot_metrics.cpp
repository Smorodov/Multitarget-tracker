#include "mot_metrics.h"

#include "defines.h"
#include "HungarianAlg.h"

#include <algorithm>
#include <cmath>
#include <unordered_map>

namespace
{
const float kForbiddenCost = 1e6f; // infeasible pair for the LAP solver
const float kUnassignableCost = 1e9f; // forbidden cell in the ID cost matrix

///
double IoU(const cv::Rect& a, const cv::Rect& b)
{
    const double x1 = std::max<double>(a.x, b.x);
    const double y1 = std::max<double>(a.y, b.y);
    const double x2 = std::min<double>(a.x + a.width, b.x + b.width);
    const double y2 = std::min<double>(a.y + a.height, b.y + b.height);
    const double inter = std::max(0.0, x2 - x1) * std::max(0.0, y2 - y1);
    const double uni = static_cast<double>(a.area()) + b.area() - inter;
    return uni > 0.0 ? inter / uni : 0.0;
}

///
/// \brief Clamp a box to the frame, drop it when it becomes empty.
///
bool ClampBox(cv::Rect& box, int width, int height)
{
    if (width <= 0 || height <= 0)
        return box.width > 0 && box.height > 0;

    const int x1 = std::max(0, box.x);
    const int y1 = std::max(0, box.y);
    const int x2 = std::min(width, box.x + box.width);
    const int y2 = std::min(height, box.y + box.height);
    if (x2 <= x1 || y2 <= y1)
        return false;
    box = cv::Rect(x1, y1, x2 - x1, y2 - y1);
    return true;
}

typedef std::map<int, std::map<int, cv::Rect>> Tracks; // id -> (frame -> box)

void Insert(Tracks& tracks, const MotDetection& det)
{
    tracks[det.id][det.frame] = det.box;
}

template<typename Vec>
std::vector<int> SortedKeys(const Vec& from)
{
    std::vector<int> keys;
    keys.reserve(from.size());
    for (const auto& kv : from)
        keys.push_back(kv.first);
    return keys; // map iteration is already sorted
}
} // namespace

double MotMetrics::Mota() const
{
    if (numObjects == 0)
        return 0.0;
    const double errs = static_cast<double>(numMisses + numFalsePositives + numSwitches);
    return 1.0 - errs / static_cast<double>(numObjects);
}

double MotMetrics::Motp() const
{
    if (numMatches == 0)
        return 0.0;
    return sumMatchDist / static_cast<double>(numMatches);
}

double MotMetrics::MotpOverlap() const
{
    if (numMatches == 0)
        return 0.0;
    return 1.0 - Motp();
}

double MotMetrics::Idf1() const
{
    const long long denom = numObjects + numPredictions;
    if (denom == 0)
        return 0.0;
    return 2.0 * static_cast<double>(idtp) / static_cast<double>(denom);
}

MotMetrics EvaluateMotSequence(const std::vector<MotDetection>& gtIn,
                               const std::vector<MotDetection>& resIn,
                               const MotEvalOptions& options)
{
    MotMetrics out;

    // 1. Split the GT by class role and clamp everything to the frame.
    Tracks gtEval, gtIgnore, hyp;
    for (const auto& d : gtIn)
    {
        MotDetection det = d;
        if (!ClampBox(det.box, options.imageWidth, options.imageHeight))
            continue;

        if (!options.evalClasses.empty() && options.evalClasses.find(det.cls) == std::end(options.evalClasses))
        {
            if (options.ignoreClasses.find(det.cls) != std::end(options.ignoreClasses))
                Insert(gtIgnore, det);
            continue; // a class that is neither evaluated nor ignored
        }
        if (options.evalClasses.empty() &&
            options.ignoreClasses.find(det.cls) != std::end(options.ignoreClasses))
        {
            Insert(gtIgnore, det);
            continue;
        }
        Insert(gtEval, det);
        ++out.numObjects;
    }

    std::map<int, std::map<int, cv::Rect>> hypByFrame; // frame -> id -> box
    Tracks hypTracks;                                 // id -> frame -> box
    for (const auto& d : resIn)
    {
        MotDetection det = d;
        if (!ClampBox(det.box, options.imageWidth, options.imageHeight))
            continue;
        if (options.minResArea > 0 &&
            static_cast<long long>(det.box.width) * det.box.height < options.minResArea)
            continue;
        ++out.numPredictions;
        hypByFrame[det.frame][det.id] = det.box;
        Insert(hypTracks, det);
    }

    // The union of frames: the GT (evaluated and ignored) and the results.
    std::set<int> frames;
    for (const auto& t : gtEval)
        for (const auto& fb : t.second)
            frames.insert(fb.first);
    for (const auto& t : gtIgnore)
        for (const auto& fb : t.second)
            frames.insert(fb.first);
    for (const auto& fb : hypByFrame)
        frames.insert(fb.first);
    out.numFrames = static_cast<long long>(frames.size());

    // Per-track frame presence for the ID measures.
    std::unordered_map<int, int> oc; // gt id -> frames present
    for (const auto& t : gtEval)
        oc[t.first] = static_cast<int>(t.second.size());
    std::unordered_map<int, int> hc; // hyp id -> frames present
    for (const auto& fb : hypByFrame)
        for (const auto& tb : fb.second)
            ++hc[tb.first];

    // Frames with IoU >= threshold per (gt, hyp) pair: the ID "tp" counts.
    // Geometric, independent of the CLEAR matching (as in py-motmetrics).
    std::map<std::pair<int, int>, long long> tps;
    for (const auto& gtTrack : gtEval)
    {
        for (const auto& hypTrack : hypTracks)
        {
            long long cnt = 0;
            const auto& gframes = gtTrack.second;
            for (const auto& fb : hypTrack.second)
            {
                auto it = gframes.find(fb.first);
                if (it == std::end(gframes))
                    continue;
                if (IoU(it->second, fb.second) >= options.minIou)
                    ++cnt;
            }
            if (cnt > 0)
                tps[{ gtTrack.first, hypTrack.first }] = cnt;
        }
    }

    // 2. CLEAR MOT, frame by frame.
    std::map<int, int> lastMatch; // gt id -> last matched hyp id (persists)
    const std::vector<int> frameList(std::begin(frames), std::end(frames));
    AssignmentProblemSolver solver;

    for (int f : frameList)
    {
        std::vector<std::pair<int, cv::Rect>> gts;  // (gt id, box)
        std::vector<std::pair<int, cv::Rect>> hyps; // (hyp id, box)
        std::vector<cv::Rect> ignores;
        for (const auto& t : gtEval)
        {
            auto it = t.second.find(f);
            if (it != std::end(t.second))
                gts.emplace_back(t.first, it->second);
        }
        for (const auto& t : gtIgnore)
        {
            auto it = t.second.find(f);
            if (it != std::end(t.second))
                ignores.push_back(it->second);
        }
        auto hIt = hypByFrame.find(f);
        if (hIt != std::end(hypByFrame))
            for (const auto& tb : hIt->second)
                hyps.emplace_back(tb.first, tb.second);

        const size_t nG = gts.size();
        const size_t nH = hyps.size();

        // Distance matrix: 1 - IoU when IoU >= minIou, forbidden otherwise.
        distMatrix_t dist(nG * nH, kForbiddenCost);
        auto distAt = [&](size_t i, size_t j) -> float& { return dist[i * nH + j]; };
        for (size_t i = 0; i < nG; ++i)
        {
            for (size_t j = 0; j < nH; ++j)
            {
                const double iou = IoU(gts[i].second, hyps[j].second);
                if (iou >= options.minIou)
                    distAt(i, j) = static_cast<float>(1.0 - iou);
            }
        }

        std::vector<bool> gtUsed(nG, false), hypUsed(nH, false);

        // 2a. Continuity: keep the matches of the previous frames.
        for (size_t i = 0; i < nG; ++i)
        {
            auto lm = lastMatch.find(gts[i].first);
            if (lm == std::end(lastMatch))
                continue;
            for (size_t j = 0; j < nH; ++j)
            {
                if (hypUsed[j] || hyps[j].first != lm->second)
                    continue;
                if (distAt(i, j) < kForbiddenCost)
                {
                    gtUsed[i] = hypUsed[j] = true;
                    out.numMatches++;
                    out.sumMatchDist += distAt(i, j);
                }
                break;
            }
        }

        // 2b. Hungarian over the remaining pairs.
        if (!gtUsed.empty() || !hypUsed.empty())
        {
            std::vector<int> freeG, freeH;
            for (size_t i = 0; i < nG; ++i)
                if (!gtUsed[i])
                    freeG.push_back(static_cast<int>(i));
            for (size_t j = 0; j < nH; ++j)
                if (!hypUsed[j])
                    freeH.push_back(static_cast<int>(j));

            if (!freeG.empty() && !freeH.empty())
            {
                const size_t nr = freeG.size(), nc = freeH.size();
                // The solver indexes the flat matrix as [row + nOfRows * col]
                // (column-major, see HungarianAlg.cpp).
                distMatrix_t sub(nr * nc);
                for (size_t i = 0; i < nr; ++i)
                    for (size_t j = 0; j < nc; ++j)
                        sub[i + nr * j] = distAt(freeG[i], freeH[j]);

                assignments_t assignment;
                solver.Solve(sub, nr, nc, assignment, AssignmentProblemSolver::optimal);

                for (size_t i = 0; i < nr && i < assignment.size(); ++i)
                {
                    const int j = assignment[i];
                    if (j < 0 || j >= static_cast<int>(nc))
                        continue;
                    if (sub[i + nr * j] >= kForbiddenCost)
                        continue; // infeasible pairing, both stay unmatched
                    const size_t gi = freeG[i], hj = freeH[j];
                    gtUsed[gi] = hypUsed[hj] = true;
                    out.numMatches++;
                    out.sumMatchDist += distAt(gi, hj);

                    auto lm = lastMatch.find(gts[gi].first);
                    if (lm != std::end(lastMatch) && lm->second != hyps[hj].first)
                        out.numSwitches++;
                    lastMatch[gts[gi].first] = hyps[hj].first;
                }
            }
        }

        // 2c. Unmatched GT -> misses (FN).
        for (size_t i = 0; i < nG; ++i)
            if (!gtUsed[i])
                out.numMisses++;

        // 2d. Unmatched hyps -> FPs, suppressed inside the ignore regions:
        // IoU >= minIou with an ignore box, or the hyp center lies inside it
        // (ignore regions are often large areas; the plain IoU test would
        // never trigger for small boxes inside them).
        for (size_t j = 0; j < nH; ++j)
        {
            if (hypUsed[j])
                continue;
            const cv::Rect hb = hyps[j].second;
            const bool suppressed = std::any_of(std::begin(ignores), std::end(ignores),
                [&](const cv::Rect& ig)
                {
                    if (IoU(hb, ig) >= options.minIou)
                        return true;
                    const int cx = hb.x + hb.width / 2;
                    const int cy = hb.y + hb.height / 2;
                    return ig.x <= cx && cx < ig.x + ig.width &&
                           ig.y <= cy && cy < ig.y + ig.height;
                });
            suppressed ? ++out.numSuppressed : ++out.numFalsePositives;
        }
    }

    // 3. ID measures: global min-cost assignment (py-motmetrics layout).
    {
        const auto gids = SortedKeys(oc);
        const auto hids = SortedKeys(hc);
        const size_t nG = gids.size(), nH = hids.size();
        const size_t n = nG + nH;

        std::unordered_map<int, int> gtIdx, hypIdx;
        for (size_t i = 0; i < nG; ++i)
            gtIdx[gids[i]] = static_cast<int>(i);
        for (size_t j = 0; j < nH; ++j)
            hypIdx[hids[j]] = static_cast<int>(j);

        // cost = fpmatrix + fnmatrix:
        //   real gt x real hyp: oc + hc - 2 * tps
        //   real gt x own dummy col: oc   (unmatched gt -> FN)
        //   own dummy row x real hyp: hc  (unmatched hyp -> FP)
        //   dummy x dummy: 0
        // Rows: [real gt 0..nG-1, dummy rows nG..nG+nH-1 (one per hyp)]
        // Cols: [real hyp 0..nH-1, dummy cols nH..nH+nG-1 (one per gt)]
        // Forbidden cells keep kUnassignableCost: a real gt may leave only
        // through its own dummy column, a real hyp only through its own
        // dummy row.
        // The solver indexes the flat matrix as [row + nOfRows * col]
        // (column-major, see HungarianAlg.cpp).
        distMatrix_t cost(n * n, kUnassignableCost);
        auto at = [&](size_t r, size_t c) -> float& { return cost[r + n * c]; };
        for (size_t r = nG; r < n; ++r)
            for (size_t c = nH; c < n; ++c)
                at(r, c) = 0.f; // dummy-dummy block

        for (size_t r = 0; r < nG; ++r)
            at(r, nH + r) = static_cast<float>(oc[gids[r]]);
        for (size_t c = 0; c < nH; ++c)
            at(nG + c, c) = static_cast<float>(hc[hids[c]]);
        for (size_t r = 0; r < nG; ++r)
        {
            for (size_t c = 0; c < nH; ++c)
            {
                const long long ex = tps.count({ gids[r], hids[c] }) ? tps[{ gids[r], hids[c] }] : 0;
                at(r, c) = static_cast<float>(oc[gids[r]] + hc[hids[c]] - 2 * ex);
            }
        }

        if (n > 0)
        {
            assignments_t assignment;
            solver.Solve(cost, n, n, assignment, AssignmentProblemSolver::optimal);
            for (size_t r = 0; r < n && r < assignment.size(); ++r)
            {
                const int c = assignment[r];
                if (c < 0 || c >= static_cast<int>(n))
                    continue;
                if (at(r, c) >= kUnassignableCost)
                    continue; // forbidden cell, treat both as unmatched

                if (r < nG && c < nH)
                {
                    const long long ex = tps.count({ gids[r], hids[c] }) ? tps[{ gids[r], hids[c] }] : 0;
                    out.idfp += hc[hids[c]] - ex;
                    out.idfn += oc[gids[r]] - ex;
                }
                else if (r < nG)
                    out.idfn += oc[gids[r]]; // unmatched gt
                else if (c < nH)
                    out.idfp += hc[hids[c]]; // unmatched hyp
            }
        }
        out.idtp = out.numObjects - out.idfn;
    }

    return out;
}
