classdef DualQPSolver < handle
    properties 
        n
        m
        A %index of active constraints
        act %status of all constraints
        q
        maxIter
    end
    
    methods
        function obj = DualQPSolver()
            obj.n=0;
            obj.m=0;
            obj.maxIter = 50;
            obj.q = 0;
        end
        
        function [x,f,output] = solve(obj, G, a, C, bl, bu, xl, xu)
            skipStep1 = false;
            [x,f,u] = obj.init(G,a,C,bl,bu,xl,xu); %step 0
            
            for it=1:obj.maxIter
                %step 1
                if (~skipStep1)
                    [p,status] = obj.selectViolatedConstraint(x);
                    if p==0
                        output = SolverStatus.Success;
                        return
                    end
                    if status == ActivationStatus.Lower
                        if p<=obj.m
                            np = C(:,p);
                        else
                            np = zeros(obj.n,1); np(p-obj.m) = 1;
                        end
                    else
                        if p<=obj.m
                            np = -C(:,p);
                        else
                            np = zeros(obj.n,1); np(p-obj.m) = -1;
                        end
                    end
                    u = [u;0];
                end

                %step 2
                [z,r] = obj.computeStep(np);
                %disp(['z=' num2str(z')])
                %disp(['r=' num2str(r')])
                [t1,t2,l] = obj.computeStepLength(p,status,x,u,z,r);
                t = min(t1,t2);

                if t==Inf
                    output = SolverStatus.Infeasible;
                    return
                end

                if t2==Inf
                    u = u+t*[-r;1];
                    u=obj.drop(l,u);
                    skipStep1 = true;
                else
                    x = x + t*z;
                    f = f + t*(z'*np)*(0.5*t + u(end));
                    u = u + t*[-r;1];
                    if t==t2
                        obj.add(p,np,status)
                        skipStep1 = false;
                    else
                        u = obj.drop(l,u);
                        skipStep1 = true;
                    end
                end
            end
            output = SolverStatus.MaxIter;
        end
        
        function [x,f,u] = init(obj,G,a,C,bl,bu,xl,xu)
            assert(size(a,2)==1);
            assert(size(bl,2)==1);
            assert(isempty(bu) || size(bu,2)==1);
            assert(isempty(xl) || size(xl,2)==1);
            assert(isempty(xu) || size(xu,2)==1);
            
            obj.n = length(a);
            
            assert(size(G,2)==obj.n);
            assert(size(C,1)==obj.n);
            
            obj.m = size(C,2);
            
            [x,f,u] = obj.init_(G,a,C,bl,bu,xl,xu);
        end
    end %methods
end