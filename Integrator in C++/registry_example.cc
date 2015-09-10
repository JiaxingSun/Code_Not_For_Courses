#include <iostream>
#include <string>
#include <vector>
#include "simpleRegistry.h"

class Animal {
public:
    void print() const {
        std::cout << "My " << getType() << " is called " << name_ << std::endl;
    }
    virtual const std::string getType() const { return "Unknown"; }
protected:
    Animal(const std::string &name) : name_(name) {}
private:
    const std::string name_;           // animal's name
};

struct Bear : Animal {
    Bear(const std::string &name) : Animal(name) { }
    virtual const std::string getType() const { return "bear"; }
};

struct Cat : Animal {
    Cat(const std::string &name) : Animal(name) { }
    virtual const std::string getType() const { return "cat"; }
};

struct Dog : Animal {
    Dog(const std::string &name) : Animal(name) { }
    virtual const std::string getType() const { return "dog"; }
};

/*
 * An object which, when instantiated, registers an Animal
 */
struct RegisterAnimal {
  RegisterAnimal(const std::string &species, // species of animal
                 Registry::Factory factory   // factory function for animal
                )
    {
      Registry::getInstance().declare(species, factory);
    }
};

/*
 * A factory function for Animals
 */
template<typename T>
Animal *makeAnimal(const std::string &name) {
    return new T(name);
}

/************************************************************************************************************/

static RegisterAnimal instance1("dog", makeAnimal<Dog>);
static RegisterAnimal instance2("cat", makeAnimal<Cat>);
static RegisterAnimal instance3("grizzly", makeAnimal<Bear>);

/************************************************************************************************************/

void printSpecies()
{
  std::cout << "Species I know about:";
  std::vector<const std::string *> species = Registry::getInstance().species();
  for (std::vector<const std::string *>::iterator ptr = species.begin(); ptr != species.end(); ++ptr) {
    std::cout << " " << **ptr;
  }
  std::cout << std::endl;
}

int main() {
    printSpecies();

    Animal *a = Registry::getInstance().lookup("cat")("Tiddles");
    a->print();

    Registry::getInstance().lookup("grizzly")("Grumpy")->print();
}
