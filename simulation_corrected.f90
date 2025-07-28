subroutine simulation(N,n_step,nu,d_time,V_in,V_out,phi0,phi1)
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
real(8), intent(in)  ::  nu
real(8), intent(in)  ::  d_time
real(8), intent(in)  ::  V_in(1,1)
real(8), intent(in)  ::  V_out(1,1)
real(8), intent(in)  ::  phi0(N,1)
real(8), intent(out)  ::  phi1(N,1)
integer(4) k,endd
real(8)  dx, t_time
real(8)  phi(N,1),phiXP(N-2,1), phiXM(N-2,1)

! parameters
dx=1.0/N ! spacing size in x direction
endd=N

t_time=0
phi=phi0

! solve the Burgers equation with FDM under explicit time scheme
do k=1,n_step
   t_time=t_time+d_time
   ! matrix-based solver

   phi1(1,:) = phi(1,:)+d_time*((-phi(1,:)*phi(2,:)+phi(1,:)*phi(1,:))/dx+(nu/dx/dx)*(phi(3,:)+phi(1,:)-2*phi(2,:)) + V_in(1,:)) ! update phi_start
   phi1(N,:) = phi(N,:)+d_time*((-phi(N,:)*phi(N,:)+phi(N,:)*phi(N-1,:))/dx+(nu/dx/dx)*(phi(N,:)+&
   &phi(N-2,:)-2*phi(N-1,:))+V_out(1,:)) ! update phi_end
   phiXP(1:endd,1)=phi(3:endd,1) ! right neighbor
   phiXM(1:endd,1)=phi(1:endd-2,1) ! left neighbor
   phi1(2:endd-1,:)=phi(2:endd-1,:)+d_time*((-phi(2:endd-1,:)*phiXP+phi(2:endd-1,:)*phiXM)/(2*dx)+(nu/dx/dx)*(phiXP+phiXM-&
   &2*phi(2:endd-1,:))) ! update phi_mid
   
   phi=phi1
enddo
phi1=phi

end subroutine

