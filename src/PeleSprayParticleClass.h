#ifndef PELESPRAYPARTICLECLASS_H
#define PELESPRAYPARTICLECLASS_H

class PeleSprayParticle
{
private:
public:
  double x[3];
  double u[3];
  double t;
  double d;
  int id;

  PeleSprayParticle() { id = 0; }
};
#endif
