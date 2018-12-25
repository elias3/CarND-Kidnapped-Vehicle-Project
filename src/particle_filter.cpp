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


	num_particles = 10;

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
	is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate)
{
	default_random_engine gen;
	for (int i = 0; i < num_particles; ++i)
	{
		if (yaw_rate != 0){
			auto theta = particles[i].theta;

			particles[i].x += (velocity / yaw_rate) * (sin(theta + delta_t * yaw_rate) - sin(theta));
			particles[i].y += (velocity / yaw_rate) * (-cos(theta + delta_t * yaw_rate) + cos(theta));
			particles[i].theta += yaw_rate * delta_t;
			
			normal_distribution<double> dist_x(particles[i].x, std_pos[0]);
			normal_distribution<double> dist_y(particles[i].y, std_pos[1]);
			normal_distribution<double> dist_theta(particles[i].theta, std_pos[2]);

			particles[i].x = dist_x(gen);
			particles[i].y = dist_y(gen);
			particles[i].theta = dist_theta(gen);
		}
	}
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> &predicted, std::vector<LandmarkObs> &observations)
{
	for (int p = 0; p < predicted.size(); ++p)
	{
		auto min = std::numeric_limits<double>::max();
		auto index = 0;
		predicted[p].id = -1;
		for (int o = 0; o < observations.size(); ++o)
		{
			auto d = dist(predicted[p].x, predicted[p].y, observations[o].x, observations[o].y);
			if (d < min)
			{
				min = d;
				index = o;
				predicted[p].id = observations[o].id;
			}
		}
	}
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
								   const std::vector<LandmarkObs> &observations, const Map &map_landmarks)
{
	for (int i = 0; i < num_particles; ++i)
	{
		auto x_part = particles[i].x;
		auto y_part = particles[i].y;
		auto theta_part = particles[i].theta;

		// transform to map coordinates
		auto transformed_observations = transform(observations, x_part, y_part, theta_part);
		auto in_range = convert_to_landmarks(x_part, y_part, sensor_range, map_landmarks);
		dataAssociation(transformed_observations, in_range);

		std::vector<int> associations;
		std::vector<double> sense_x;
		std::vector<double> sense_y;

		auto total_weight = 1.0;
		for (int o = 0; o < transformed_observations.size(); ++o)
		{
			if (transformed_observations[o].id != -1)
			{
				associations.push_back(transformed_observations[o].id);
				sense_x.push_back(transformed_observations[o].x);
				sense_y.push_back(transformed_observations[o].y);
			}
			LandmarkObs landmark;
			bool is_found = false;
			for (int j = 0; j < in_range.size(); j++)
			{
				if (transformed_observations[o].id == in_range[j].id)
				{
					landmark = in_range[j];
					is_found = true;
					break;
				}
			}
			if (is_found)
			{
				auto w = weight(std_landmark[0], std_landmark[1], transformed_observations[o].x, transformed_observations[o].y, landmark.x, landmark.y);
				total_weight *= w;
			}
		}
		particles[i].weight = total_weight;
		SetAssociations(particles[i], associations, sense_x, sense_y);
	}
}

void ParticleFilter::resample()
{
	std::random_device rd;
	std::mt19937 gen(rd());

	for (int i = 0; i < num_particles; ++i)
	{
		weights[i] = particles[i].weight;
	}
	std::discrete_distribution<> d(weights.begin(), weights.end());
	std::vector<Particle> particles_new;
	for (int i = 0; i < num_particles; ++i)
	{
		particles_new.push_back(particles[d(gen)]);
	}
	particles = particles_new;
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
