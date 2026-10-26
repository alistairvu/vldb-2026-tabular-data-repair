#include <nlohmann/json.hpp>

#include <algorithm>
#include <cstddef>
#include <iostream>
#include <iterator>
#include <regex>
#include <stdexcept>
#include <string>

#include "manifest.hpp"

using json = nlohmann::json;

int main() {
    try {
        const auto manifest = json::parse(
            embedded_manifest, embedded_manifest + embedded_manifest_size);
        const std::string input(std::istreambuf_iterator<char>(std::cin), {});
        const auto request = json::parse(input);
        if (!request.is_object() || request.size() != 2 ||
            !request.contains("column") || !request.at("column").is_string() ||
            !request.contains("candidates") || !request.at("candidates").is_array()) {
            throw std::runtime_error(
                "request must contain only string column and array candidates");
        }

        const auto& column = request.at("column").get_ref<const std::string&>();
        const auto& candidates = request.at("candidates");
        if (!std::all_of(candidates.begin(), candidates.end(),
                         [](const auto& value) { return value.is_string(); })) {
            throw std::runtime_error("candidates must contain only strings");
        }

        const auto& rules = manifest.at("oracles");
        const auto rule_iterator = rules.find(column);
        if (rule_iterator == rules.end())
            throw std::runtime_error("unknown column: " + column);
        const auto& rule = rule_iterator.value();

        json answers = json::array();
        if (rule.at("type") == "enum") {
            const auto& allowed = rule.at("values");
            for (const auto& item : candidates) {
                const auto& value = item.get_ref<const std::string&>();
                answers.push_back(value != "" && value != "?" &&
                                  std::find(allowed.begin(), allowed.end(), item) != allowed.end());
            }
        } else {
            const std::regex expression(rule.at("regex").get<std::string>(),
                                        std::regex::ECMAScript);
            for (const auto& item : candidates) {
                const auto& value = item.get_ref<const std::string&>();
                answers.push_back(value != "" && value != "?" &&
                                  std::regex_match(value, expression));
            }
        }
        std::cout << answers.dump();
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "oracle: " << error.what() << '\n';
        return 2;
    }
}
