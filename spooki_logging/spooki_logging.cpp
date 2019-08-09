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
// #include "spooki_utility/pipes/PipeUtilities.hpp"

#include <iostream>
#include <string>
#include <memory>
#include <mutex>
#include <ostream>
#include <fstream>




#include <boost/shared_ptr.hpp>

#include <boost/version.hpp>

/*
 * BOOST LOG Includes
 * Version thing: empty_deleter.hpp is for building with older versions of boost
 * and should be replaced with the null_deleter object from null_deleter.hpp.
 */
#if BOOST_VERSION == 105400
#include <boost/log/utility/empty_deleter.hpp>
#else
#include <boost/core/null_deleter.hpp>
#endif

#ifndef BOOST_LOG_DYN_LINK
#warning "BOOST LOG DYN LINK IS UNDEFINED"
#endif

#include <boost/log/sinks.hpp>
#include <boost/log/expressions.hpp>
#include <boost/log/attributes.hpp>
#include <boost/log/support/date_time.hpp>

#include "spooki_logging/spooki_logging.hpp"

const pst::log_severity_level SPOOKI_DEFAULT_LOG_LEVEL = pst::log_severity_level::info;
const bool SPOOKI_DEFAULT_LOG_TIMESTAMPS = true;

namespace expr = boost::log::expressions;
namespace sinks = boost::log::sinks;
namespace attrs = boost::log::attributes;
namespace src = boost::log::sources;
namespace keywords = boost::log::keywords;

namespace pst
{

std::unique_ptr<spooki_logging> spooki_logging::_instance = nullptr; // initialize pointer

log_severity_level get_severity_level_from_env()
{
    log_severity_level level = SPOOKI_DEFAULT_LOG_LEVEL;
    char *env_var = getenv("SPOOKI_LOG_LEVEL");

    if(env_var) {
        std::istringstream iss(env_var);
        try {
            level = log_severity_str_to_enum(iss.str());
        } catch (std::exception e) {
            std::cerr << "Environment variable SPOOKI_LOG_LEVEL is set but could not be converted to valid log level" << std::endl
                << "The value is " << iss.str() << " and possible values are : " << std::endl
                << std::endl << pst::possible_log_levels() << std::endl;
        }
    }

    return level;
}

bool get_show_timestamps_from_env(){

    char *env_var = getenv("SPOOKI_LOG_TIMESTAMPS");
    bool show_timestamps = SPOOKI_DEFAULT_LOG_TIMESTAMPS;

    if(env_var){
        std::istringstream iss(env_var);
        iss >> show_timestamps;
    }

    return (bool) show_timestamps;
}

log_severity_level log_severity_str_to_enum(std::string enum_name){
    size_t nb_members = sizeof(log_severity_level_names) / sizeof(*log_severity_level_names);

    for(size_t i = 0; i < nb_members ; ++i){
        if(strcmp(log_severity_level_names[i], enum_name.c_str()) == 0){
            return (log_severity_level) i;
        }
    }

    throw std::exception();
}

std::string possible_log_levels(){
    size_t nb_members = sizeof(log_severity_level_names) / sizeof(*log_severity_level_names);

    std::ostringstream oss;

    oss << "Possible values for SPOOKI_LOG_LEVEL" << std::endl;
    for(size_t i = 0; i < nb_members ; ++i){
        oss << "    " << (log_severity_level) i << std::endl;
    }

    return oss.str();
}

spooki_logging::spooki_logging() :
        logs_initialized(false), fatal_error_log_use_count(0), error_log_use_count(0)
{
}

spooki_logging::~spooki_logging()
{
    report_special();
}

const std::unique_ptr<spooki_logging> & spooki_logging::Instance()
{
    if(!_instance){
        _instance.reset(new spooki_logging());
        _instance->setup_log_sinks("/dev/null");
        _instance->set_log_entry_format(true);
    };
    return _instance;
}

/******************************************************************************
 * This function sets up the _textSink object of the spooki_logging object.
 *
 * This sink takes text and has two streams attached to it: the
 *****************************************************************************/
void spooki_logging::setup_log_sinks(std::string filename){
    _textSink.reset(new text_sink);
    {
        text_sink::locked_backend_ptr pBackend = _textSink->locked_backend();
#if BOOST_VERSION == 105400
        boost::shared_ptr< std::ostream > pStream(&std::clog, boost::log::v2_mt_posix::empty_deleter());
#else
        boost::shared_ptr< std::ostream > pStream(&std::clog, boost::null_deleter());
#endif
        pBackend->add_stream(pStream);
        boost::shared_ptr< std::ofstream > pStream2(new std::ofstream(filename));
        assert(pStream2->is_open());
        pBackend->add_stream(pStream2);
    }
    boost::log::core::get()->add_sink(_textSink);
}


/******************************************************************************
 * Sets the severity level required for messages to be allowed to be logged
 *****************************************************************************/
void spooki_logging::set_severity_filer_level(log_severity_level severity_filter_level){
    _log_severity_filter_level = severity_filter_level;
    _textSink->set_filter(
            expr::attr< log_severity_level >("Severity").or_default(log_severity_level::normal) >= severity_filter_level
    // This following line demonstrates things you can do with an optional tag attribute
    // || expr::begins_with(expr::attr< std::string >("Tag").or_default(std::string()), "IMPORTANT")
    );
}

log_severity_level spooki_logging::get_current_severity_filter_level(){
    return _log_severity_filter_level;
}

void spooki_logging::set_show_timestamps(bool with_timestamps){

    set_log_entry_format(with_timestamps);
}


/******************************************************************************
 * Creates the boost::log::expressions expression to specify the format of the
 * log entries and some of the logic as to which attributes to show when.
 *
 * Note the named scope attribute.  This is cool to use with the
 *
 * BOOST_LOG_NAMED_SCOPE("A NAMED SCOPE FOR LOGGING")
 *
 * which adds that attribute and creates an object whose destructor will remove
 * that attribute thus log entries will have this attribute for the duration
 * of the scope where the macro was used.
 *****************************************************************************/
void spooki_logging::set_log_entry_format(bool with_timestamps){
    _textSink->set_formatter(expr::stream
                                     // << expr::attr< unsigned int >("RecordID")
                                     << expr::if_(expr::has_attr("TimeStamp"))
                                     [
                                             //expr::stream << expr::attr< std::string >("Tag")
                                             expr::stream << "[" << expr::format_date_time< boost::posix_time::ptime >("TimeStamp", "%H:%M:%S.%f") << "]"
                                     ]
                                     << " [" << expr::attr< log_severity_level >("Severity") << "] "
                                     // << " [" << expr::attr< boost::posix_time::time_duration >("Uptime") << "] "
                                     << expr::if_(expr::has_attr("Scope"))
                                     [
                                             //expr::stream << expr::attr< std::string >("Tag")
                                             expr::stream << "[" << expr::format_named_scope("Scope", keywords::format = "%n", keywords::iteration = expr::reverse) << "]"
                                     ]
                                     << expr::smessage); // here goes the log record text

    // Uncomment the line with "RecordID in the expression
    // attrs::counter< unsigned int > RecordID(1);
    // boost::log::core::get()->add_global_attribute("RecordID", RecordID);

    if(with_timestamps){
        attrs::local_clock TimeStamp;
        boost::log::core::get()->add_global_attribute("TimeStamp", TimeStamp);
    }

    // Uncomment the timer line in the expression to get this in the log records
    // attrs::timer Timer;
    // boost::log::core::get()->add_global_attribute("Uptime", Timer);

#ifdef DEBUG
    /*
     * This named scope attribute is really interesting, use the
     * BOOST_LOG_NAMED_SCOPE("A NAMED SCOPE FOR LOGGING")
     *
     * Maybe this could be used only for debugging and we could have
     */
    attrs::named_scope Scope;
    boost::log::core::get()->add_thread_attribute("Scope", Scope);
#endif
}

void spooki_logging::log_special(std::string msg)
{
    special_messages.push_back(msg);
}

void spooki_logging::report_special()
{
    for(auto it = special_messages.begin(); it != special_messages.end(); ++it)
    {
        std::cerr << *it << std::endl;
    }
}

size_t & spooki_logging::errorCount()
{
    return error_log_use_count;
}

size_t & spooki_logging::fatalErrorCount()
{
    return fatal_error_log_use_count;
}

spooki_logging_scope_sentry::spooki_logging_scope_sentry(log_severity_level new_level){
    auto &inst = spooki_logging::Instance();
    _initial = inst->get_current_severity_filter_level();
    inst->set_severity_filer_level(new_level);
}

spooki_logging_scope_sentry::~spooki_logging_scope_sentry(){
    spooki_logging::Instance()->set_severity_filer_level(_initial);
}

}
