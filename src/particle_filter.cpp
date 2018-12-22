/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h>
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[])
{
	default_random_engine gen;

	normal_distribution<double> dist_x(x, std[0]);
	normal_distribution<double> dist_y(y, std[1]);
	normal_distribution<double> dist_theta(theta, std[2]);

	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of
	//   x, y, theta and their uncertainties from GPS) and all weights to 1.
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).

	num_particles = 100; // TODO: Set the number of particles

	for (int i = 0; i < num_particles; ++i)
	{
		Particle particle;
		particle.id = i;
		particle.x = dist_x(gen);
		particle.y = dist_y(gen);
		particle.theta = dist_theta(gen);

		weights.push_back(1.0);
		particles.push_back(particle);
		// Print your samples to the terminal.
		cout << "Particle  " << i << " " << particle.x << " " << particle.y << " " << particle.theta << endl;
	}
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate)
{
	// std::sin(pi / 6)
	default_random_engine gen;
	for (int i = 0; i < num_particles; ++i)
	{
		normal_distribution<double> dist_x(particles[i].x, std_pos[0]);
		normal_distribution<double> dist_y(particles[i].y, std_pos[1]);
		normal_distribution<double> dist_theta(particles[i].theta, std_pos[2]);

		auto theta = particles[i].theta;
		particles[i].x += (velocity / yaw_rate) * (sin(theta + delta_t * yaw_rate) - sin(theta)) + dist_x(gen);
		particles[i].y += (velocity / yaw_rate) * (-cos(theta + delta_t * yaw_rate) + cos(theta)) + dist_y(gen);
		particles[i].theta += yaw_rate * delta_t + dist_theta(gen);
	}
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> &predicted, std::vector<LandmarkObs> &observations)
{
	for (int p = 0; p < predicted.size(); ++p)
	{
		auto min = std::numeric_limits<double>::max();
		for (int o = 0; o < observations.size(); ++o)
		{
			auto d = dist(predicted[p].x, predicted[p].y, observations[o].x, observations[o].y);
			if (d > min)
			{
				min = d;
				predicted[p].id = observations[o].id;
			}
		}
	}
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to
	//   implement this method and use it as a helper during the updateWeights phase.
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
								   const std::vector<LandmarkObs> &observations, const Map &map_landmarks)
{
	for (int i = 0; i < num_particles; ++i)
	{
		auto x_part = particles[i].x;
		auto y_part = particles[i].y;

		std::vector<LandmarkObs> transformed_observations;
		// transform to map coordinates
		for (int o = 0; o < observations.size(); ++o)
		{
			auto theta = particles[i].theta;
			auto x_obs = observations[o].x;
			auto y_obs = observations[o].y;
			LandmarkObs l;
			l.x = x_part + (cos(theta) * x_obs) - (sin(theta) * y_obs);
			l.y = y_part + (sin(theta) * x_obs) + (cos(theta) * y_obs);
			transformed_observations.push_back(l);
		}

		std::vector<LandmarkObs> in_range;
		for (int l = 0; l < map_landmarks.landmark_list.size(); ++l)
		{
			auto &landmark = map_landmarks.landmark_list[l];
			auto d = dist(landmark.x_f, landmark.y_f, x_part, y_part);
			if (d <= sensor_range)
			{
				LandmarkObs observed;
				observed.id = landmark.id_i;
				observed.x = landmark.x_f;
				observed.y = landmark.y_f;
				in_range.push_back(observed);
			}
		}
		dataAssociation(transformed_observations, in_range);

		std::vector<int> associations;
		std::vector<double> sense_x;
		std::vector<double> sense_y;

		auto total_weight = 1;
		for (int o = 0; o < transformed_observations.size(); ++o)
		{
			associations.push_back(transformed_observations[o].id);
			sense_x.push_back(transformed_observations[o].x);
			sense_y.push_back(transformed_observations[o].y);

			LandmarkObs landmark;
			for (int j = 0; j < in_range.size(); j++)
			{
				if (transformed_observations[o].id == in_range[j].id)
				{
					landmark = in_range[j];
					break;
				}
			}
			total_weight *= weight(std_landmark[0], std_landmark[1], transformed_observations[o].x, transformed_observations[o].y, landmark.x, landmark.y);
		}
		particles[i].weight = total_weight;
		cout << "Particle " << i << " weight " << particles[i].weight << endl;
		SetAssociations(particles[i], associations, sense_x, sense_y);
		// particles[i].associations = transformed_observations;
	}
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html
}

void ParticleFilter::resample()
{
	std::random_device rd;
	std::mt19937 gen(rd());
	std::discrete_distribution<> d(weights.begin(), weights.end());
	std::vector<Particle> particles_new;
	for (int i = 0; i < num_particles; ++i)
	{
		particles_new.push_back(particles[d(gen)]);
	}
	// TODO: Resample particles with replacement with probability proportional to their weight.
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
}

void ParticleFilter::SetAssociations(Particle &particle, const std::vector<int> &associations,
									 const std::vector<double> &sense_x, const std::vector<double> &sense_y)
{
	//particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
	// associations: The landmark id that goes along with each listed association
	// sense_x: the associations x mapping already converted to world coordinates
	// sense_y: the associations y mapping already converted to world coordinates
	particle.associations = associations;
	particle.sense_x = sense_x;
	particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
	copy(v.begin(), v.end(), ostream_iterator<int>(ss, " "));
	string s = ss.str();
	s = s.substr(0, s.length() - 1); // get rid of the trailing space
	return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
	copy(v.begin(), v.end(), ostream_iterator<float>(ss, " "));
	string s = ss.str();
	s = s.substr(0, s.length() - 1); // get rid of the trailing space
	return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
	copy(v.begin(), v.end(), ostream_iterator<float>(ss, " "));
	string s = ss.str();
	s = s.substr(0, s.length() - 1); // get rid of the trailing space
	return s;
}
