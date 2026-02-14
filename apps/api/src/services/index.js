/**
 * Service index — instantiates all services with their dependencies.
 *
 * Usage:
 *   const services = require('./services')({ repos, config, models });
 *   await services.auth.login({ email, password });
 */
const AuthService = require('./AuthService');
const UserService = require('./UserService');
const MemberService = require('./MemberService');
const TrainerService = require('./TrainerService');
const AdminService = require('./AdminService');

function createServices({ repos, config, models }) {
  return {
    auth: new AuthService({ repos, config }),
    user: new UserService({ repos }),
    member: new MemberService({ repos }),
    trainer: new TrainerService({ repos }),
    admin: new AdminService({ repos, models }),
  };
}

module.exports = createServices;
