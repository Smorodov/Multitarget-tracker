#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "mot_metrics.h"

namespace
{
///
std::vector<std::string> SplitCsvLine(const std::string& line)
{
    std::vector<std::string> fields;
    std::stringstream ss(line);
    std::string field;
    while (std::getline(ss, field, ','))
        fields.push_back(field);
    return fields;
}

///
/// \brief Load a MOTChallenge-format file (GT or results).
///
bool LoadFile(const std::string& path, bool needClass, std::vector<MotDetection>& out)
{
    std::ifstream file(path);
    if (!file.is_open())
    {
        std::cerr << "mot_eval: cannot open " << path << std::endl;
        return false;
    }

    std::string line;
    int lineNo = 0;
    while (std::getline(file, line))
    {
        ++lineNo;
        if (line.empty() || line[0] == '#')
            continue;
        const auto fields = SplitCsvLine(line);
        if (fields.size() < 6)
        {
            std::cerr << "mot_eval: " << path << ":" << lineNo << ": expected >= 6 fields, got "
                      << fields.size() << std::endl;
            return false;
        }
        MotDetection det;
        try
        {
            det.frame = std::stoi(fields[0]);
            det.id = std::stoi(fields[1]);
            det.box.x = std::stoi(fields[2]);
            det.box.y = std::stoi(fields[3]);
            det.box.width = std::stoi(fields[4]);
            det.box.height = std::stoi(fields[5]);
        }
        catch (const std::exception&)
        {
            std::cerr << "mot_eval: " << path << ":" << lineNo << ": bad number" << std::endl;
            return false;
        }
        if (fields.size() > 6)
            det.conf = std::strtof(fields[6].c_str(), nullptr);
        if (fields.size() > 7)
            det.cls = std::atoi(fields[7].c_str());
        if (needClass && det.cls < 0)
        {
            std::cerr << "mot_eval: " << path << ":" << lineNo
                      << ": a class id (column 8) is required for this GT file" << std::endl;
            return false;
        }
        out.push_back(det);
    }
    return true;
}

///
/// \brief Map names/numbers from a class-list option to class ids.
///
/// The optional types file lists class names in the GT class id order
/// (id = line number + 1, as in the MOTChallenge/VisDrone type files).
///
bool ParseClassList(const std::string& arg, const std::map<std::string, int>& nameToId,
                    std::set<int>& out)
{
    if (arg.empty())
        return true;
    std::stringstream ss(arg);
    std::string item;
    while (std::getline(ss, item, ';'))
    {
        if (item.empty())
            continue;
        char* end = nullptr;
        const long val = std::strtol(item.c_str(), &end, 10);
        if (end && *end == '\0' && end != item.c_str())
        {
            out.insert(static_cast<int>(val));
        }
        else
        {
            auto it = nameToId.find(item);
            if (it == std::end(nameToId))
            {
                std::cerr << "mot_eval: unknown class name \"" << item
                          << "\" (check --types or use numeric ids)" << std::endl;
                return false;
            }
            out.insert(it->second);
        }
    }
    return true;
}

void PrintReport(const std::string& label, const MotMetrics& m)
{
    std::printf("Sequence: %s\n", label.c_str());
    std::printf("  frames            %lld\n", m.numFrames);
    std::printf("  GT objects        %lld\n", m.numObjects);
    std::printf("  predictions       %lld\n", m.numPredictions);
    std::printf("  matches           %lld\n", m.numMatches);
    std::printf("  FN (misses)       %lld\n", m.numMisses);
    std::printf("  FP                %lld\n", m.numFalsePositives);
    std::printf("  FP suppressed     %lld\n", m.numSuppressed);
    std::printf("  IDSW              %lld\n", m.numSwitches);
    std::printf("  IDTP/IDFP/IDFN    %lld / %lld / %lld\n", m.idtp, m.idfp, m.idfn);
    std::printf("  MOTA              %.4f\n", m.Mota());
    std::printf("  MOTP (1-IoU)      %.4f\n", m.Motp());
    std::printf("  MOTP (IoU)        %.4f\n", m.MotpOverlap());
    std::printf("  IDF1              %.4f\n", m.Idf1());
}
} // namespace

int main(int argc, char** argv)
{
    std::string gtPath, resPath, typesPath, csvPath, label = "sequence";
    std::string gtClassesArg, ignoreClassesArg;
    float minIou = 0.5f;
    int width = 0, height = 0;
    long long minArea = 0;

    for (int i = 1; i < argc; ++i)
    {
        std::string arg = argv[i];
        if (arg.rfind("--", 0) != 0)
        {
            std::cerr << "mot_eval: unexpected argument " << arg << std::endl;
            return 1;
        }
        std::string name = arg, val;
        const size_t eq = arg.find('=');
        if (eq != std::string::npos)
        {
            name = arg.substr(0, eq);
            val = arg.substr(eq + 1);
        }
        else
        {
            name = arg;
            val = (i + 1 < argc) ? argv[++i] : "";
        }
        // Strip a lone "--name" form to its bare name for comparison.
        if (name == "--gt")
            gtPath = val;
        else if (name == "--res")
            resPath = val;
        else if (name == "--types")
            typesPath = val;
        else if (name == "--gt_classes")
            gtClassesArg = val;
        else if (name == "--gt_ignore_classes")
            ignoreClassesArg = val;
        else if (name == "--iou")
            minIou = std::strtof(val.c_str(), nullptr);
        else if (name == "--width")
            width = std::atoi(val.c_str());
        else if (name == "--height")
            height = std::atoi(val.c_str());
        else if (name == "--min_area")
            minArea = std::atoll(val.c_str());
        else if (name == "--out_csv")
            csvPath = val;
        else if (name == "--label")
            label = val;
        else
        {
            std::cerr << "mot_eval: unknown option " << arg << std::endl;
            return 1;
        }
    }

    if (gtPath.empty() || resPath.empty())
    {
        std::cerr << "Usage: mot_eval --gt <gt.txt> --res <res.txt>"
                  " [--types <types.txt>] [--gt_classes names;..] [--gt_ignore_classes names;..]"
                  " [--iou 0.5] [--width W --height H] [--min_area px^2]"
                  " [--label seq] [--out_csv res.csv]" << std::endl;
        return 1;
    }

    // Optional types file: names in the GT class id order (id = line + 1).
    std::map<std::string, int> nameToId;
    if (!typesPath.empty())
    {
        std::ifstream file(typesPath);
        if (!file.is_open())
        {
            std::cerr << "mot_eval: cannot open " << typesPath << std::endl;
            return 1;
        }
        std::string line;
        int id = 0;
        while (std::getline(file, line))
        {
            ++id;
            while (!line.empty() && (line.back() == '\r' || line.back() == ' '))
                line.pop_back();
            if (!line.empty())
                nameToId[line] = id;
        }
    }

    std::vector<MotDetection> gt, res;
    if (!LoadFile(gtPath, true, gt) || !LoadFile(resPath, false, res))
        return 1;

    MotEvalOptions opts;
    opts.minIou = minIou;
    opts.imageWidth = width;
    opts.imageHeight = height;
    opts.minResArea = minArea;
    if (!ParseClassList(gtClassesArg, nameToId, opts.evalClasses) ||
        !ParseClassList(ignoreClassesArg, nameToId, opts.ignoreClasses))
        return 1;

    const MotMetrics m = EvaluateMotSequence(gt, res, opts);
    PrintReport(label, m);

    if (!csvPath.empty())
    {
        bool writeHeader = true;
        {
            std::ifstream check(csvPath);
            writeHeader = !check.is_open() || check.peek() == std::ifstream::traits_type::eof();
        }
        std::ofstream csv(csvPath, std::ios::app);
        if (!csv.is_open())
        {
            std::cerr << "mot_eval: cannot open " << csvPath << std::endl;
            return 1;
        }
        if (writeHeader)
            csv << "label,mota,motp,motp_overlap,idf1,idsw,fp,fn,suppressed,"
                   "idtp,idfp,idfn,num_objects,num_predictions,matches,frames" << std::endl;
        csv << label << ','
            << m.Mota() << ',' << m.Motp() << ',' << m.MotpOverlap() << ',' << m.Idf1() << ','
            << m.numSwitches << ',' << m.numFalsePositives << ',' << m.numMisses << ','
            << m.numSuppressed << ',' << m.idtp << ',' << m.idfp << ',' << m.idfn << ','
            << m.numObjects << ',' << m.numPredictions << ',' << m.numMatches << ','
            << m.numFrames << std::endl;
    }

    return 0;
}
