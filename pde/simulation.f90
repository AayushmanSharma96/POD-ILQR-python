subroutine simulation(N,n_step,gamma,d_time,T,h,phi0,phi1)
!----------------------------------------------------------------------
! Subroutine to calculate order parameter with Allen-Cahn Equation
! N: state dimension (integer)           [input]
! n_step: amount of time steps (int)     [input]
! gamma: gradient energy coefficient     [input]
! d_time: time step (sec)                [input]
! T: one of control variables (N,N)      [input]
! h: the other control variable (N,N)    [input]
! phi0: initial order parameter (N,N)    [input]
! phi1: final order parameter matrix     [output]
! ---------------------------------------------------------------------

implicit none
integer(4), intent(in)  ::  N
integer(4), intent(in)  ::  n_step
real(8), intent(in)  ::  gamma
real(8), intent(in)  ::  d_time
real(8), intent(in)  ::  T(N,N)
real(8), intent(in)  ::  h(N,N)
real(8), intent(in)  ::  phi0(N,N)
real(8), intent(out)  ::  phi1(N,N)
integer(4) k,endd
real(8)  dx, dy, t_time
real(8)  phi(N,N),phiXP(N,N), phiXM(N,N), phiYP(N,N), phiYM(N,N)

! parameters
dx=1.0/N ! spacing size in x direction
dy=dx ! spacing size in y direction
endd=N

t_time=0
phi=phi0
! solve the Allen-Cahn equation with FDM under explicit time scheme
do k=1,n_step
   t_time=t_time+d_time
   ! matrix-based solver
   phiXP(:,1:endd-1)=phi(:,2:endd) ! right neighbor
   phiXP(:,endd)=phi(:,1) ! right neighbor with periodical boundary condition
   phiXM(:,2:endd)=phi(:,1:endd-1) ! left neighbor
   phiXM(:,1)=phi(:,endd) ! left neighbor on left boundary
   phiYP(1:endd-1,:)=phi(2:endd,:) ! upper neighbor
   phiYP(endd,:)=phi(1,:) ! upper neighbor on top boundary
   phiYM(2:endd,:)=phi(1:endd-1,:) ! lower neighbor
   phiYM(1,:)=phi(endd,:) ! lower neighbor on bottom boundary
   phi1=phi+d_time*(-4*phi*phi*phi-2*T*phi-h+gamma*(phiXP+phiXM+phiYP+phiYM-4*phi)/dx/dx) ! update phi 
   phi=phi1
enddo
phi1=phi

end subroutine

