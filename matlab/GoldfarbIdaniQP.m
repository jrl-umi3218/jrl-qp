classdef GoldfarbIdaniQP < DualQPSolver
    properties
        G,
        a,
        C,
        b
        L
        R
        J
    end
    
    methods
        function obj = GoldfarbIdaniQP()
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
            
            obj.L = chol(G)';
            obj.R = zeros(obj.q,obj.q);
            obj.J = inv(obj.L');
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
            d = obj.J'*np;
            z = obj.J(:,obj.q+1:end)*d(obj.q+1:end);
            r = obj.R\d(1:obj.q);
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
            obj.act(k) = ActivationStatus.Inactive;
            
            %update R and J
            obj.R = obj.R(:,[1:(l-1),(l+1):end]);
            for i=l:(obj.q2)
                Qi = givens(obj.R(i,i),obj.R(i+1,i));
                obj.R([i,i+1],i:end) = Qi*obj.R([i,i+1],i:end);
                obj.J(:,[i,i+1]) = obj.J(:,[i,i+1])*Qi';
            end
            obj.q = obj.q - 1;
        end
        
        function add(obj,p,status)
            obj.A = [obj.A, p];
            obj.act(p) = status;
            
            %update R and J
            np = obj.C(:,p);
            d = obj.J'*np; %Todo: duplicate with computeStep
            for i=obj.n-1:-1:obj.q+1
                Qi = givens(d(i),d(i+1));
                d(i:i+1) = Qi*d(i:i+1);
                obj.J(:,[i,i+1]) = obj.J(:,[i,i+1])*Qi';
            end
            obj.R = [[obj.R; zeros(1,obj.q)] d(1:obj.q+1)];
            obj.q = obj.q + 1;
        end
    end
end