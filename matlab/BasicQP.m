classdef BasicQP < DualQPSolver
    properties
        G,
        a,
        C,
        b
    end
    
    methods
        function obj = BasicQP()
            obj@DualQPSolver();
        end
        
        function [x,f,u] = init_(obj,G,a,C,bl,bu,xl,xu)
            
            assert(isempty(bu));
            assert(isempty(xl));
            assert(isempty(xu));
            obj.G = G;
            obj.a = a;
            obj.C = C;
            obj.b = bl;
            
            x = -G\a;
            f = 0.5*a'*x;
            u = zeros(0,1);
            obj.A = zeros(1,0);
            obj.act = repmat([ActivationStatus.Inactive],1,obj.m);
        end
        
        function [p,status] = selectViolatedConstraint(obj, x)
            s = obj.C'*x - obj.b;
            [~,p] = min(s);
            if s(p)<0 && obj.act(p)==ActivationStatus.Inactive
                status = ActivationStatus.Lower;
            else
                p=0;
                status = ActivationStatus.Inactive;
            end
        end
        
        function [z,r] = computeStep(obj,np)
            y = [obj.G -obj.C(:,obj.A); obj.C(:,obj.A)' zeros(obj.q)]\[np;zeros(obj.q,1)];
            z = y(1:obj.n);
            r = -y(obj.n+(1:obj.q));
        end
        
        function [t1,t2,l] = computeStepLength(obj,p,status,x,u,z,r)
            t1 = Inf;
            l = 0;
            for i=1:obj.q
                if r(i)>0
                    ti = u(i)/r(i);
                    if ti < t1
                        t1 = ti;
                        l = i;
                    end
                end
            end
            if norm(z)>1e-14
                t2 = (obj.b(p)-obj.C(:,p)'*x)/(z'*obj.C(:,p));
            else
                t2 = Inf;
            end
        end
        
        function u = drop(obj,l,u)
            k = obj.A(l);
            obj.A = obj.A([1:(l-1),(l+1):end]);
            u = u([1:(l-1),(l+1):end]);
            obj.q = obj.q - 1;
            obj.act(k) = ActivationStatus.Inactive;
        end
        
        function add(obj,p,np,status)
            obj.A = [obj.A, p];
            obj.q = obj.q + 1;
            obj.act(p) = status;
        end
    end
end