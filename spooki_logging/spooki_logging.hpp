//----------------------------------------------------------------------------------------
//
//   ENVIRONNEMENT CANADA
//   Centre Meteorologique Canadien
//   2121 route Transcanadienne
//   Dorval, Quebec
//   H9P 1J3
//
//   Date :  05-december-2007
//----------------------------------------------------------------------------------------
#ifndef SPOOKI_LOGGING_HPP_
#define SPOOKI_LOGGING_HPP_

#include <string>
#include <cstring>
#include <iostream>
#include <ostream>
#include <sstream>
#include <vector>

// required for BOOST_LOG_SEV() to work
#include <boost/log/sources/record_ostream.hpp>
#include <boost/log/sources/severity_logger.hpp>
#include <boost/log/sinks.hpp>

/*
 * To remove, this is just for the places where I haven't removed references to boost log trivial
 */
//#include <boost/log/trivial.hpp>

/*
 * To remove, this is for all the places that use expressions for filters
 */
//#include <boost/log/expressions.hpp>

namespace pst
{

    /*
     * The severity level and the function to convert them to strings
     * NOTE: This works based on the matching order of the enum elements
     * and the strings in the char* array.
     */
    enum log_severity_level
    {
        debug,
        normal,
        special,
        info,
        warning,
        error,
        fatal
    };

    static const char* const log_severity_level_names[] =
    {
            "debug",
            "normal",
            "special",
            "info",
            "warning",
            "error",
            "fatal"
    };

    /*
     * Overload operator<<() so that putting a log_severity_level in a stream object
     * will output the corresponding string.  This allows the logging stuff to output
     * the right tag in the log records.
     */
    template< typename CharT, typename TraitsT >
    inline std::basic_ostream< CharT, TraitsT >&
    operator<< ( std::basic_ostream< CharT, TraitsT >& strm, log_severity_level lvl)
    {

        if (static_cast< std::size_t >(lvl) < (sizeof(log_severity_level_names) / sizeof(*log_severity_level_names)))
            strm << log_severity_level_names[lvl];
        else
            strm << static_cast< int >(lvl);
        return strm;
    }

    log_severity_level log_severity_str_to_enum(std::string enum_name);

    log_severity_level get_severity_level_from_env();

    std::string possible_log_levels();

    bool get_show_timestamps_from_env();

    class spooki_logging
    {
    public:

        virtual ~spooki_logging();
        static const std::unique_ptr<spooki_logging> &Instance();

        boost::log::sources::severity_logger<log_severity_level> _boost_severity_logger;

        log_severity_level get_current_severity_filter_level();
        void set_severity_filer_level(log_severity_level severity_filter_level);
        void set_show_timestamps(bool with_timestamps);

        size_t & fatalErrorCount();
        size_t & errorCount();

        void log_special(std::string msg);
        void report_special();
        std::vector<std::string> special_messages;

    private:
        log_severity_level _log_severity_filter_level;
        typedef boost::log::sinks::synchronous_sink< boost::log::sinks::text_ostream_backend > text_sink;

        // See implementation of setup_log_sinks for why using boost::shared_ptr instead of std::shared_ptr
        boost::shared_ptr<text_sink>  _textSink;
        std::pair<boost::log::attribute_set::iterator, bool> timestamp_attribute;
        void setup_log_sinks(std::string filename);
        void set_log_entry_format(bool with_timestamps);

        spooki_logging(void);
        spooki_logging(const spooki_logging & rhs);
        spooki_logging & operator=(const spooki_logging & rhs);

        static std::unique_ptr<spooki_logging> _instance;

        bool logs_initialized;
        size_t fatal_error_log_use_count;
        size_t error_log_use_count;
    };
    class spooki_logging_scope_sentry
    {
    public:
        spooki_logging_scope_sentry(log_severity_level new_level);
        ~spooki_logging_scope_sentry();
        log_severity_level _initial;
    };
}

#define SCOPED_LOG_SEVERITY(level) pst::spooki_logging_scope_sentry BOOST_PP_CAT(_scoped_log_sentry_, __LINE__) (level);

#ifndef THIS_FILE
#define THIS_FILE ( strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__ )
#endif

#ifndef LOG_WHERE
#define LOG_WHERE " WITHIN FILE " << THIS_FILE << " NEAR LINE : " << __LINE__ << " "
#endif

#ifndef BOOST_LOG_BLANK
#define BOOST_LOG_BLANK std::cerr << std::endl;
#endif

#ifndef BOOST_LOG_DEBUG
#define BOOST_LOG_DEBUG BOOST_LOG_SEV(pst::spooki_logging::Instance()->_boost_severity_logger, pst::log_severity_level::debug) << __PRETTY_FUNCTION__ << " : " << __LINE__ << "] "
#endif

#ifndef BOOST_LOG_INFO
#define BOOST_LOG_INFO \
BOOST_LOG_SEV(pst::spooki_logging::Instance()->_boost_severity_logger, pst::log_severity_level::info)
#endif

#ifndef BOOST_LOG_SPECIAL
#define BOOST_LOG_SPECIAL(MSG) \
do{\
    auto &inst = pst::spooki_logging::Instance();\
    std::ostringstream oss;\
    oss << MSG;\
    std::string message = oss.str();\
    message = pst::string_utilities::colorizeString(pst::string_utilities::YELLOW, message);\
    BOOST_LOG_SEV(inst->_boost_severity_logger, pst::log_severity_level::special) << message;\
    inst->log_special(message);\
} while(0);
#endif

#ifndef BOOST_LOG_WARNING
#define BOOST_LOG_WARNING \
BOOST_LOG_SEV(pst::spooki_logging::Instance()->_boost_severity_logger, pst::log_severity_level::warning)
#endif

#ifndef BOOST_LOG_ERROR
#define BOOST_LOG_ERROR(MSG)\
do{\
    auto &inst = pst::spooki_logging::Instance();\
    ++inst->errorCount();\
    std::ostringstream oss;\
    oss << MSG;\
    std::string message = oss.str();\
    message = pst::string_utilities::colorizeString(pst::string_utilities::RED, message);\
    BOOST_LOG_SEV(inst->_boost_severity_logger, pst::log_severity_level::error)\
        << std::endl << std::endl << "    " << message;\
} while(0);
#endif

#ifndef BOOST_LOG_FATAL
#define BOOST_LOG_FATAL(MSG)\
do{\
    auto &inst = pst::spooki_logging::Instance();\
    ++inst->fatalErrorCount();\
    std::ostringstream oss;\
    oss << MSG << std::endl;\
    std::string message = oss.str();\
    message = pst::string_utilities::colorizeString(pst::string_utilities::RED, message);\
    BOOST_LOG_SEV(inst->_boost_severity_logger, pst::log_severity_level::fatal) << message;\
} while(0);
#endif


#endif /*SPOOKI_LOGGING_HPP_*/
