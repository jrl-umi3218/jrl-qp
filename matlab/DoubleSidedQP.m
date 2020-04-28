classdef DoubleSidedQP < DualQPSolver
    properties
        G,
        a,
        C,
        bl
        bu
        xl
        xu
        L
        R
        J
        me
        nb %number of bound
    end
    
    methods
        function obj = DoubleSidedQP()
            obj@DualQPSolver();
        end
        
        function [x,f,u] = init_(obj,G,a,C,bl,bu,xl,xu)
            assert(isempty(xl) || length(xl) == obj.n);
            assert(length(xl) == length(xu));
            assert(all(bl<=bu));
            obj.G = G;
            obj.a = a;
            obj.C = C;
            obj.bl = bl;
            obj.bu = bu;
            obj.xl = xl;
            obj.xu = xu;
            
            if isempty(xl)
                obj.nb = 0;
                obj.act = repmat([ActivationStatus.Inactive],1,obj.m);
            else
                obj.nb = obj.n;
                obj.act = repmat([ActivationStatus.Inactive],1,obj.m+obj.n);
            end
            
            obj.A = zeros(1,0);
            for i=1:obj.m
                if bl(i)==bu(i)
                    obj.A = [obj.A i];
                    obj.act(i) = ActivationStatus.Equality;
                    obj.q = obj.q+1;
                end
            end
            obj.me = obj.q;
            
            obj.L = chol(G)';
            [Qb,Rb] = qr(obj.L\obj.C(:,obj.A));
            obj.R = Rb(1:obj.q,1:obj.q);
            obj.J = (obj.L')\Qb;
            
            x = obj.J*[(obj.R')\obj.bl(obj.A); -obj.J(:,obj.q+1:end)'*a];
            f = x'*(0.5*G*x +a);
            u = obj.R\(obj.J(:,1:obj.q)'*(G*x+a));
        end
        
        function [p,status] = selectViolatedConstraint(obj, x)
            smin = 0;
            p=0;
            status = ActivationStatus.Inactive;
            for i=1:obj.m
                if obj.act(i) == ActivationStatus.Inactive
                    sl = obj.C(:,i)'*x - obj.bl(i);
                    su = obj.bu(i) - obj.C(:,i)'*x;
                    if sl<smin
                        smin = sl;
                        p = i;
                        status = ActivationStatus.Lower;
                    end
                    if su<smin
                        smin = su;
                        p = i;
                        status = ActivationStatus.Upper;
                    end
                end
            end
            for i=1:obj.nb
                if obj.act(obj.m+i) == ActivationStatus.Inactive
                    sl = x(i) - obj.xl(i);
                    su = obj.xu(i) - x(i);
                    if sl<smin
                        smin = sl;
                        p = obj.m+i;
                        status = ActivationStatus.Lower;
                    end
                    if su<smin
                        smin = su;
                        p = obj.m+i;
                        status = ActivationStatus.Upper;
                    end
                end
            end
        end
        
        function [z,r] = computeStep(obj,np)
            d = obj.J'*np;
            z = obj.J(:,obj.q+1:end)*d(obj.q+1:end);
            r = obj.R\d(1:obj.q);
            obj.log(end).J = obj.J;
            obj.log(end).R = obj.R;
            obj.log(end).d = d';
        end
        
        function [t1,t2,l] = computeStepLength(obj,p,status,x,u,z,r)
            t1 = Inf;
            l = 0;
            for i=(obj.me+1):obj.q
                if r(i)>0
                    ti = u(i)/r(i);
                    if ti < t1
                        t1 = ti;
                        l = i;
                    end
                end
            end
            if norm(z)>1e-14
                if status == ActivationStatus.Lower
                    if p<=obj.m
                        t2 = (obj.bl(p)-obj.C(:,p)'*x)/(z'*obj.C(:,p));
                    else
                        t2 = (obj.xl(p-obj.m)-x(p-obj.m))/z(p-obj.m);
                    end
                else
                    if p<=obj.m
                        t2 = (obj.bu(p)-obj.C(:,p)'*x)/(z'*obj.C(:,p));
                    else
                        t2 = (obj.xu(p-obj.m)-x(p-obj.m))/z(p-obj.m);
                    end
                end
            else
                t2 = Inf;
            end
        end
        
        function u = drop(obj,l,u)
            k = obj.A(l);
            obj.A = obj.A([1:(l-1),(l+1):end]);
            u = u([1:(l-1),(l+1):end]);
            obj.act(k) = ActivationStatus.Inactive;
            obj.q = obj.q - 1;
            
            %update R and J
            obj.R = obj.R(:,[1:(l-1),(l+1):end]);
            for i=l:obj.q
                Qi = givens(obj.R(i,i),obj.R(i+1,i));
                obj.R([i,i+1],i:end) = Qi*obj.R([i,i+1],i:end);
                obj.J(:,[i,i+1]) = obj.J(:,[i,i+1])*Qi';
            end
            obj.R = obj.R(1:obj.q,1:obj.q);
        end
        
        function add(obj,p,np,status)
            obj.A = [obj.A, p];
            obj.act(p) = status;
            
            %update R and J
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