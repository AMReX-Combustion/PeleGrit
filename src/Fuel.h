#ifndef FUEL_H
#define FUEL_H

class Fuel{
  private:
  const double nasapolyair_[5]={1052.71406, -0.3745355, 8.361477e-4, -3.32111e-7, -4.683905e-11};

  const double nasapolydecane_[6]={3.19882239e1, 4.77244922e-2, -1.62276391e-5, 
                                   2.50963259e-9, -1.45215772e-13, -4.66392840e4};
  public:
  const double MW_decane_ = 0.1422817; //kg/mol
  const double MW_air_ = 0.0289647; //kg/mol

  const double atm_ = 1.01325e5;

  const double R_ = 8.314462618;

  double Tref_;

  double boilT_;

  double Cpl_;

  double href_;

  double latentref_;

  const double rhod_=0.64; //kg/m3

  Fuel(){};

  void updateh()
  {
    href_=calcEnthalpy(Tref_, nasapolydecane_, MW_decane_);
  };

  double readNasapolyair(size_t i){return nasapolyair_[i];}

  double readNasapolydecane(size_t i){return nasapolydecane_[i];}

  double calcEnthalpy(const double T, const double* nasapoly, const double mw) 
  {
    return (nasapoly[0] + nasapoly[1]*T/2.0 + nasapoly[2]*T*T/3.0
           + nasapoly[3]*T*T*T/4.0 + nasapoly[4]*std::pow(T,4.0)/5.0  
           + nasapoly[5]/T)*R_/mw*T;
  }
};
#endif
