#include <Kokkos_Core.hpp>
#include "GritPeleSprayInterface.h"
#include "Spray_Pele.h"
#include "Yarn.h"
#include "LagrangeInterp.h"
#include "GaussianDeposit.h"

#define MASTER  0

void GritPeleSprayInterface::initializeGrit() 
{
  Kokkos::initialize();
  ptr_spray = std::unique_ptr<PeleSprayParticle[]>(new PeleSprayParticle[Spray_Pele::NDUST]);
}

void GritPeleSprayInterface::advanceSprays(const long int NP, const double dt, 
                                      const double* dx, const int DIM, 
                                      const int NX, const int NY, const int NZ,
                                      const int rhogindex, const int rhogUgindex,
                                      const int rhogegindex, const int Tgindex, 
                                      const int Ygindex, bool do_move,
                                      const int sNX, const int sNY, const int sNZ) 
{
  assert(Spray_Pele::NDUST>=NP);

  //order of interpolation and deposition
  constexpr int NH = 0;

  const double pi = 4.0*atan(1.0);

  const double part_dt = 0.5*dt;

  Spray_Pele grit_spray;

  //copy from Amr to Grit
  auto xdhost = Kokkos::create_mirror_view(grit_spray.loc);
  auto udhost = Kokkos::create_mirror_view(grit_spray.vel);
  auto shost  = Kokkos::create_mirror_view(grit_spray.state);
  auto dhost  = Kokkos::create_mirror_view(grit_spray.dia);
  auto Thost  = Kokkos::create_mirror_view(grit_spray.temp);

  for(size_t n=0; n<NP; n++) 
  {
    if(ptr_spray[n].id > -1) 
    {    
      shost(n) = 1;
    }
    else 
    {
      shost(n) = 0;
      break;
    }
 
    for(size_t l=0; l<DIM; l++) 
    {
      xdhost(n,l) = ptr_spray[n].x[l];
      udhost(n,l) = ptr_spray[n].u[l];
    }
    dhost(n) = ptr_spray[n].d;
    Thost(n) = ptr_spray[n].t;
  }

  Kokkos::deep_copy(grit_spray.loc, xdhost);
  Kokkos::deep_copy(grit_spray.vel, udhost);
  Kokkos::deep_copy(grit_spray.state, shost);
  Kokkos::deep_copy(grit_spray.dia, dhost);
  Kokkos::deep_copy(grit_spray.temp, Thost);

  //Mesh size
  const int NG = NX*NY*NZ;

  Kokkos::View<double**> Ug("Ug", NG, DIM); 
  Kokkos::View<double*> rhog("rhog", NG);
  Kokkos::View<double*> Tg("Tg", NG);
  Kokkos::View<double*> Yg("Yg", NG);

  //all variables from Pele have cgs units
  auto Ughost = Kokkos::create_mirror_view(Ug);
  auto rhoghost = Kokkos::create_mirror_view(rhog);
  auto Tghost = Kokkos::create_mirror_view(Tg);
  auto Yghost = Kokkos::create_mirror_view(Yg);

  for(int n=0; n<NG; n++) 
  {
    int index_rhoamr = n+rhogindex*NG;
    int index_Tamr = n+Tgindex*NG;
    int index_Yamr = n+Ygindex*NG;
    rhoghost(n) = *(ptr_state+index_rhoamr);
    Tghost(n) = *(ptr_state+index_Tamr);
    Yghost(n) = *(ptr_state+index_Yamr);
    for(int l=0; l<DIM; l++) 
    {
      int index_rhoUamr = n+(rhogUgindex+l)*NG;
      Ughost(n,l) = (*(ptr_state + index_rhoUamr))/rhoghost(n);
    }
  }

  Kokkos::deep_copy(Ug, Ughost);
  Kokkos::deep_copy(rhog, rhoghost);
  Kokkos::deep_copy(Tg, Tghost);
  Kokkos::deep_copy(Yg, Yghost);

  //Interpolation

  Spray_Pele::Vectr3PointType Ugpar("Ugpar");
  Spray_Pele::ScalarPointType rhogpar("rhogpar");
  Spray_Pele::ScalarPointType Tgpar("Tgpar");
  Spray_Pele::ScalarPointType Ygpar("Ygpar");
  Spray_Pele::ScalarPointType Pgpar("Pgpar");

  if(DIM==3)
  {
    LagrangeInterp<NH>::interpolateVector(NX-1, NY-1, NZ-1, DIM, NP, Ug, grit_spray.loc, grit_spray.state, Ugpar);
    LagrangeInterp<NH>::interpolateScalar(NX-1, NY-1, NZ-1, NP, rhog, grit_spray.loc, grit_spray.state, rhogpar);
    LagrangeInterp<NH>::interpolateScalar(NX-1, NY-1, NZ-1, NP, Tg, grit_spray.loc, grit_spray.state, Tgpar);
    LagrangeInterp<NH>::interpolateScalar(NX-1, NY-1, NZ-1, NP, Yg, grit_spray.loc, grit_spray.state, Ygpar);
  }
  else
  {
    LagrangeInterp<NH>::interpolateVector(NX-1, NY-1, DIM, NP, Ug, grit_spray.loc, grit_spray.state, Ugpar);
    LagrangeInterp<NH>::interpolateScalar(NX-1, NY-1, NP, rhog, grit_spray.loc, grit_spray.state, rhogpar);
    LagrangeInterp<NH>::interpolateScalar(NX-1, NY-1, NP, Tg, grit_spray.loc, grit_spray.state, Tgpar);
    LagrangeInterp<NH>::interpolateScalar(NX-1, NY-1, NP, Yg, grit_spray.loc, grit_spray.state, Ygpar);
  }

  Spray_Pele::ScalarPointType Tv("Tv");
  Spray_Pele::ScalarPointType hvatTd("hvatTd");
  Spray_Pele::ScalarPointType hfg("hfg");
  Spray_Pele::ScalarPointType Yv("Yv");
  Spray_Pele::ScalarPointType Cpv("Cpv");
  Spray_Pele::ScalarPointType Cpvf("Cpvf");
  Spray_Pele::ScalarPointType Sp_Bm("Sp_Bm");

  //fueldata is hard coded for decane NC10H22
  const double T_boil = fueldata.boilT_;
  const double Cp_l = fueldata.Cpl_; //cgs unit
  const double Tref = fueldata.Tref_;
  const double latentref = fueldata.latentref_; //cgs unit
  const double rho_d = fueldata.rhod_; //si unit

  const double atm = fueldata.atm_;
  const double R = fueldata.R_;
  const double MWf = fueldata.MW_decane_; //si unit kg/mol
  const double MWa = fueldata.MW_air_;
  const double hvref = fueldata.href_; //si unit

  //air
  const double a0 = fueldata.readNasapolyair(0);
  const double a1 = fueldata.readNasapolyair(1);
  const double a2 = fueldata.readNasapolyair(2);
  const double a3 = fueldata.readNasapolyair(3);
  const double a4 = fueldata.readNasapolyair(4);

  //NC10H22
  const double b0 = fueldata.readNasapolydecane(0);
  const double b1 = fueldata.readNasapolydecane(1);
  const double b2 = fueldata.readNasapolydecane(2);
  const double b3 = fueldata.readNasapolydecane(3);
  const double b4 = fueldata.readNasapolydecane(4);
  const double b5 = fueldata.readNasapolydecane(5);

  Kokkos::parallel_for(NP, KOKKOS_LAMBDA(const size_t& n) 
  {
    if(grit_spray.state(n) !=1) return;
    //calculate latent heat
    double Td=grit_spray.temp(n);
    Tv(n) = grit_spray.temp(n) + 1.0/3.0*(Tgpar(n)-Td); 
    hvatTd(n) = b0 + b1*Td/2.0 + b2*Td*Td/3.0 
              + b3*Td*Td*Td/4.0 + b4*pow(Td,4.0)/5.0
              + b5/Td;
    hvatTd(n) *= R/MWf*Td;
    double hd = hvref - (latentref*1e-4) + (Cp_l*1e-4)*(Td-Tref);
    hfg(n) = hvatTd(n) - hd;

    //calc Y_v
    double MWg = (1.0-Ygpar(n))*MWa + Ygpar(n)*MWf;
    double Psat = atm*exp(hfg(n)*MWf*(1.0/R)*(1.0/T_boil-1.0/Td));
    Pgpar(n) = (rhogpar(n)*1000)*R/MWg*Tgpar(n);
    double Yd = MWf*Psat/(MWg*Pgpar(n)+(MWf-MWg)*Psat);
    if(Yd>1.0||Yd<0.0) 
    {
      grit_spray.state(n) = 0;
      return;
    }
    Yv(n) = Yd + 1.0/3.0*(Ygpar(n)-Yd);
    Sp_Bm(n) = max(1e-10,(Yd - Ygpar(n))/(1.0 - Yd + 1e-30));

    //calc Cp_v
    double Cpair = a0 + a1*Tv(n) + a2*Tv(n)*Tv(n) + a3*Tv(n)*Tv(n)*Tv(n)
                 + a4*pow(Tv(n),4);

    Cpvf(n)= (b0 + b1*Tv(n) + b2*Tv(n)*Tv(n) + b3*Tv(n)*Tv(n)*Tv(n)
                 + b4*pow(Tv(n),4))*R/MWf;

    Cpv(n) = Cpair*(1.0-Yv(n)) + Cpvf(n)*Yv(n);
  });

  //transport - harded coded for decane NC10H22
  //double Geo = 2;
  //double LJpwd = 704.917;
  //double sigma = 6.675;

  Spray_Pele::ScalarPointType lambdav("lambdav");
  Spray_Pele::ScalarPointType mu_v("mu_v");
  Spray_Pele::ScalarPointType D_v("D_v");
  Spray_Pele::ScalarPointType Sc_v("Sc_v");
  Spray_Pele::ScalarPointType Pr_v("Pr_v");

  const double lambda_A = 2.58e-5;
  const double lambda_r = 0.7;
  const double mu_b = 1.7894e-5;
  const double mu_T0 = 273.11;
  const double mu_S = 110.56;

  Kokkos::parallel_for(NP, KOKKOS_LAMBDA(const size_t& n) 
  {
    if(grit_spray.state(n) !=1) return;
    //get lambda_v, mu_v, D_v
    lambdav(n) = Cpv(n)*lambda_A*pow((Tv(n)/298.0),lambda_r);

    double mu_air = mu_b*pow(Tv(n)/mu_T0,1.5)*(mu_T0+mu_S)/(Tv(n)+mu_S);
    double mu_fuel = 1.7083e-8*Tv(n)+5.922e-8;
    mu_v(n) = mu_air*(1.0-Yv(n))+mu_fuel*Yv(n);

    double denomD = cbrt(208.56) + cbrt(20.1);
    D_v(n) = 1e-3*pow(Tv(n),1.75)*sqrt(1.0/MWa/1000+1.0/MWf/1000)/(Pgpar(n)/atm)/denomD/denomD;//cgs

    Sc_v(n) = mu_v(n)/(rhogpar(n)*1000)/(D_v(n)*1e-4);
    Pr_v(n) = Cpv(n)*mu_v(n)/lambdav(n);
  });

  Spray_Pele::Vectr3PointType rhsu("rhsu");
  Spray_Pele::Vectr3PointType srcu("srcu");
  Spray_Pele::ScalarPointType rhse("rhse");
  Spray_Pele::ScalarPointType srce("srce");
  Spray_Pele::ScalarPointType srcm("srcm");

  Spray_Pele::ScalarPointType m("m");

  //std::cout<<"integrate U"<<std::endl;
  
  double cellsizex, cellsizey, cellsizez;
  cellsizex = dx[0];
  cellsizey = dx[1];
  cellsizez = dx[2];

  //integrate U
  Kokkos::parallel_for(NP, KOKKOS_LAMBDA (const size_t& n) 
  {
    if(grit_spray.state(n) != 1) return;
    double temp=0; 
    double magU=0;
    for(size_t l=0;l<DIM;l++) {
      temp=(grit_spray.vel(n,l)-Ugpar(n,l));
      magU+=temp*temp;
    }
    magU=sqrt(magU);
    grit_spray.Re(n) = rhogpar(n)*magU*grit_spray.dia(n)/(mu_v(n)*10);
    double dragCoef, drag;
    dragCoef = 24.0/grit_spray.Re(n)*(1.0+pow(grit_spray.Re(n),2.0/3.0)/6.0);
    drag = 0.125*rhogpar(n)*dragCoef*pi*grit_spray.dia(n)*grit_spray.dia(n)*magU;
    //m is also g here
    m(n)=1.0/6.0*pi*rho_d*grit_spray.dia(n)*grit_spray.dia(n)*grit_spray.dia(n);
    for(size_t l=0; l<DIM; l++) {
      rhsu(n,l) = (Ugpar(n,l)-grit_spray.vel(n,l))*drag;
      grit_spray.vel(n,l) = grit_spray.vel(n,l) + rhsu(n,l)/m(n)*part_dt;
    }
 } );

  if(do_move) {
    Kokkos::parallel_for(NP, KOKKOS_LAMBDA (const size_t& n) {
      if(grit_spray.state(n) != 1) return;
      //integrate locations
      grit_spray.loc(n,0) = grit_spray.loc(n,0) + grit_spray.vel(n,0)*dt/cellsizex;
      grit_spray.loc(n,1) = grit_spray.loc(n,1) + grit_spray.vel(n,1)*dt/cellsizey;
      grit_spray.loc(n,2) = grit_spray.loc(n,2) + grit_spray.vel(n,2)*dt/cellsizez;
    } );
  }

  Spray_Pele::ScalarPointType Shv("Shv");

  //calculate Sherwood number and integrate m
  Kokkos::parallel_for(NP, KOKKOS_LAMBDA (const size_t& n) {
    if(grit_spray.state(n) != 1) return;
    Shv(n) = 1.0+pow(1.0+grit_spray.Re(n)*Sc_v(n),1.0/3.0);
    Shv(n) = Shv(n)*max(pow(grit_spray.Re(n),0.077),1.0);
    double fm = max(Sp_Bm(n)/(log(1+Sp_Bm(n))*pow((1+Sp_Bm(n)),0.7)),0.0);
    Shv(n) = 2.0 + (Shv(n)-2.0)*fm;
    srcm(n) = -pi*rhogpar(n)*D_v(n)*grit_spray.dia(n)*Shv(n)*log(1+Sp_Bm(n));
    m(n) = m(n) + srcm(n)*part_dt;
    if(m(n)<1e-16) { 
      grit_spray.state(n) = 0;
    }
    else {
      grit_spray.dia(n) = pow(m(n)/rho_d*6.0/pi,1.0/3.0);
    }
  } );

  //calculate Nusselt number
  Kokkos::parallel_for(NP, KOKKOS_LAMBDA (const size_t& n) {
    if(grit_spray.state(n) != 1) return;
    double Nud = 1.0 + pow(1.0+grit_spray.Re(n)*Pr_v(n),1.0/3.0)*(max(pow(grit_spray.Re(n),0.077),1.0));
    double ratio = Cpvf(n)/Cpv(n)*Pr_v(n)/Sc_v(n)*Shv(n);
    double phi = ratio/Nud;
    double Sp_Bt = pow(1.0+Sp_Bm(n),phi)-1.0;
    double log_BT, FT;

    double Nu_iter = Nud;
    size_t count = 0;
    double eps = 1e-3;
    double err =0.0;

    do {
      double BT_iter = Sp_Bt;
      log_BT = log(1.0+Sp_Bt);
      FT = Sp_Bt/(log_BT*(pow(1.0+Sp_Bt,0.7)));
      FT = min(FT,1.0);
      Nu_iter = 2.0 + (Nu_iter-2.0)*FT;
      phi = ratio/Nu_iter;
      Sp_Bt = pow(1.0+Sp_Bm(n),phi)-1.0;
      err = fabs(Sp_Bt-BT_iter)/Sp_Bt;
      count++;
    }while(err>eps||count<50);

    Sp_Bt = min(Sp_Bt,20.0);
    log_BT = log(1.0+Sp_Bt);
    FT = Sp_Bt/(log_BT*(pow(1.0+Sp_Bt,0.7)));
    FT = min(FT,1.0);
    Nud = 2.0+(Nu_iter-2.0)*FT;
    FT = min(log_BT/Sp_Bt,1.0);
    Nud = Nud*FT;

    rhse(n) = (Tgpar(n)-grit_spray.temp(n))*(lambdav(n)*1e5)*Nud*pi*grit_spray.dia(n);
    double rhsT = (rhse(n)+(hfg(n)*1e4)*srcm(n))/Cp_l/m(n);
    grit_spray.temp(n) = grit_spray.temp(n) + rhsT*part_dt;
  } );

  //deposition
  Kokkos::View<double**> Srcug("Srcug", NG, DIM);
  Kokkos::View<double*> Srcmg("Srcmg", NG);
  Kokkos::View<double*> Srceg("Srceg", NG);

  Kokkos::parallel_for(NP, KOKKOS_LAMBDA (const size_t& n) {
    if(grit_spray.state(n) != 1) return;
    double smudotu=0.0;
    double upart2=0.0;
    for(size_t l=0; l<DIM; l++) {
      srcu(n,l) = rhsu(n,l) + srcm(n)*grit_spray.vel(n,l);
      smudotu += srcu(n,l)*grit_spray.vel(n,l);
      upart2 += grit_spray.vel(n,l)*grit_spray.vel(n,l);
    }
    srce(n) = rhse(n) + smudotu + srcm(n)*((hvatTd(n)*1e4)+0.5*upart2);
  } );

  if(DIM==3)
  {
    GaussianDeposit<NH>::depositVectr3(NX-1, NY-1, NZ-1, DIM, NP, Srcug, grit_spray.loc, grit_spray.state, srcu);
    GaussianDeposit<NH>::depositScalar(NX-1, NY-1, NZ-1, NP, Srcmg, grit_spray.loc, grit_spray.state, srcm);
    GaussianDeposit<NH>::depositScalar(NX-1, NY-1, NZ-1, NP, Srceg, grit_spray.loc, grit_spray.state, srce);
  }
  else
  {
    GaussianDeposit<NH>::depositVectr3(NX-1, NY-1, DIM, NP, Srcug, grit_spray.loc, grit_spray.state, srcu);
    GaussianDeposit<NH>::depositScalar(NX-1, NY-1, NP, Srcmg, grit_spray.loc, grit_spray.state, srcm);
    GaussianDeposit<NH>::depositScalar(NX-1, NY-1, NP, Srceg, grit_spray.loc, grit_spray.state, srce);
  }

  auto suhost = Kokkos::create_mirror_view(Srcug);
  auto smhost = Kokkos::create_mirror_view(Srcmg);
  auto sehost = Kokkos::create_mirror_view(Srceg);

  Kokkos::deep_copy(suhost, Srcug);
  Kokkos::deep_copy(smhost, Srcmg);
  Kokkos::deep_copy(sehost, Srceg);

  double invV = 1.0;
  for(int l=0; l<DIM; l++)
  {
    invV *= 1.0/dx[l];
  }

  //the size of the source array != the size of the state array
  const int DIFFNX = (NX-sNX)/2;
  const int DIFFNY = (NY-sNY)/2;
  const int DIFFNZ = (NZ-sNZ)/2;

  const int sNG = sNX*sNY*sNZ;

  //copy from Grit to Pele
  for(int n=0; n<sNG; n++) {
    //give the index n of the source array
    //return the index nn of hte state array
    const int nn = getIndex(n, sNX, sNY, sNZ, DIFFNX, DIFFNY, DIFFNZ); 
    
    //source for the rho equation
    const int index_rhoamr = n + rhogindex*sNG;
    *(ptr_source + index_rhoamr)-=smhost(nn)*invV;

    //source for the Y equation
    const int index_Yamr = n + Ygindex*sNG;
    *(ptr_source + index_Yamr)-=smhost(nn)*invV;

    //source for the energy equation
    const int index_rhoeamr = n + rhogegindex*sNG;
    *(ptr_source + index_rhoeamr)-=sehost(nn)*invV;

    //source for the momentum equation
    for(int l=0; l<DIM; l++) {
      const int index_rhoUamr = n + (rhogUgindex+l)*sNG;
      *(ptr_source + index_rhoUamr)-=suhost(nn,l)*invV;
    }
  }

  Kokkos::deep_copy(shost, grit_spray.state);
  Kokkos::deep_copy(xdhost, grit_spray.loc);
  Kokkos::deep_copy(udhost, grit_spray.vel);
  Kokkos::deep_copy(dhost, grit_spray.dia);
  Kokkos::deep_copy(Thost, grit_spray.temp);

  for(size_t n=0; n<NP; n++) {
    if(shost(n)==0){
      ptr_spray[n].id = -1;
      break;
    }
    ptr_spray[n].d = dhost(n);
    ptr_spray[n].t = Thost(n);
    for(size_t l=0; l<DIM; l++) {
      ptr_spray[n].x[l] = xdhost(n,l);
      ptr_spray[n].u[l] = udhost(n,l);
    }
  }
}

int GritPeleSprayInterface::getIndex(const int sn, const int sNX, 
                                     const int sNY, const int sNZ,
                                     const int DIFFNX, const int DIFFNY, 
                                     const int DIFFNZ) {
  //i,j,k on the smaller grid
  int k = sn/(sNX*sNY);
  int j = (sn-k*sNX*sNY)/sNX;
  int i = (sn-k*sNX*sNY)%sNX;

  assert(i<sNX && j<sNY && k<sNZ);

  //absolute i,j,k on the bigger grid
  i += DIFFNX;
  j += DIFFNY;
  k += DIFFNZ;

  const int lNX = sNX + 2*DIFFNX;
  const int lNY = sNY + 2*DIFFNY;

  return k*lNX*lNY+j*lNX+i;
}

void GritPeleSprayInterface::finalizeGrit() {
  Kokkos::finalize();
}
