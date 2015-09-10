#if !defined(SIMPLE_REGISTRY_H)
#define SIMPLE_REGISTRY_H
#include <map>
#include <sstream>
#include <stdexcept>

class Animal;                           // forward declaration

/*
 * A singleton class to handle registration of Animals
 */
class Registry {
public:
    typedef Animal *(*Factory)(const std::string &name);
  /*
   * Return the (singleton) Registry object
   */
  static Registry& getInstance() {
    static Registry instance; // Guaranteed to be destroyed; Instantiated on first use.
    return instance;
  }

  /*
   * Declare an animal by species
   */
  void declare(const std::string &species, // the animal's species
               Factory factory             // a factory function to make the desired animal
              ) {
    registry_.insert(std::make_pair(species, factory));
  }

  /*
   * return the named factory function
   */
  Factory lookup(const std::string &species // the animal in question
                ) {
    MapType::iterator i = registry_.find(species);
    if (i == registry_.end()) {     // not found
      std::ostringstream os;
      os << "Unable to find " << species;
      throw std::runtime_error(os.str());
    }

    return i->second;
  }

  /*
   * Return the sorted list of species of known animals
   */
  std::vector<const std::string *> species() const {
    std::vector<const std::string *> species;
    species.reserve(registry_.size());

    for (MapType::const_iterator ptr = registry_.begin(); ptr != registry_.end(); ++ptr) {
      species.push_back(&ptr->first);
    }

    return species;
  }
private:
  Registry() {}
  ~Registry() {}
  // Not implemented --- can't be called
  Registry(const Registry&);
  void operator=(const Registry&);

  typedef std::map<std::string, Factory> MapType;
  MapType registry_; // our real registry
};
#endif
