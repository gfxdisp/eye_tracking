#ifndef HDRMFS_EYE_TRACKER_UTILS_HPP
#define HDRMFS_EYE_TRACKER_UTILS_HPP

#include <vector>
#include <string>

namespace et {

class Utils {
public:
    static std::vector<std::vector<float>>
    readFloatColumnsCsv(const std::string& filename);
    static std::vector<std::vector<float>>
    readFloatRowsCsv(const std::string& filename);
    static void writeFloatCsv(std::vector<std::vector<float>> &data, const std::string& filename);

    static std::string getCurrentTimeText();
};

} // namespace et

#endif //HDRMFS_EYE_TRACKER_UTILS_HPP
