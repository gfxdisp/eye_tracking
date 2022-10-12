#ifndef HDRMFS_EYE_TRACKER_UTILS_HPP
#define HDRMFS_EYE_TRACKER_UTILS_HPP

#include <string>
#include <vector>

namespace et {

class Utils {
public:
    /**
     * Reads the CSV file with float values in the column-major order.
     * @param filename Path to the CSV file.
     * @return Vector of columns of the CSV content.
     */
    static std::vector<std::vector<float>>
    readFloatColumnsCsv(const std::string &filename);
    /**
     * Reads the CSV file with float values in the row-major order.
     * @param filename Path to the CSV file.
     * @return Vector of rows of the CSV content.
     */
    static std::vector<std::vector<float>>
    readFloatRowsCsv(const std::string &filename);
    /**
     * Writes the float matrix to CSV file in row-major order.
     * @param data Matrix with the float data.
     * @param filename Path to CSV file to be saved.
     */
    static void writeFloatCsv(std::vector<std::vector<float>> &data,
                              const std::string &filename);

    /**
     * Converts current timestamp to a human-readable format.
     * @return A string with a converted timestamp.
     */
    static std::string getCurrentTimeText();
};

} // namespace et

#endif //HDRMFS_EYE_TRACKER_UTILS_HPP
