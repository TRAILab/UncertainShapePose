#
# This file is part of https://github.com/TRAILab/UncertainShapePose
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>
#

import scipy.stats as st

# Input @p: probability \in (0,1)
# output @c: P( |x - \mu| < c * sigma ) = p, for a gaussian distribution x \in N(mu, sigma)
def prob_to_coeff(p):
    c = st.norm.ppf((p+1)/2.0)
    return c

def pdf(x, mu, sigma):
    return st.norm.pdf(x, loc=mu, scale=sigma)

if __name__ == "__main__":
    # unit test
    a=prob_to_coeff(0.9974)
    b=prob_to_coeff(0.95)
    c=prob_to_coeff(0.68)
    print(a,b,c)


    # check pdf
    print('pdf(0,0,1)', pdf(0,0,1))
    print('pdf(1,0,1)', pdf(1,0,1))

