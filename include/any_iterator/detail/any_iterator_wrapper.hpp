//  (C) Copyright Thomas Becker 2005. Permission to copy, use, modify, sell and
//  distribute this software is granted provided this copyright notice appears
//  in all copies. This software is provided "as is" without express or implied
//  warranty, and with no claim as to its suitability for any purpose.

// Revision History
// ================
//
// 27 Dec 2006 (Thomas Becker) Created
// 12 Jul 2010 (Thomas Becker acting on bug report by Edgar Binder)
// Bug fix: Constructors and create function of <code>any_iterator_wrapper</code>
// from wrapped iterator must take their argument by const reference (performance!).

#ifndef ANY_ITERATOR_WRAPPER_01102007TMB_HPP
#define ANY_ITERATOR_WRAPPER_01102007TMB_HPP

// Includes
// ========

#include "any_iterator_abstract_base.hpp"
#include "any_iterator_metafunctions.hpp"
#include <boost/type_traits/add_const.hpp>
#include <boost/type_traits/remove_const.hpp>
#include <boost/cast.hpp>

namespace IteratorTypeErasure
{

  namespace detail
  {
  
    ///////////////////////////////////////////////////////////////////////
    // 
    // The partial specializations of any_iterator_wrapper (which is the
    // the equivalent of boost::any::holder) mirror the hierarchy of
    // boost's iterator traversal tags.
    //
    // The first four template arguments are as in boost::iterator_facade.
    // The last template argument is the traversal tag of the most
    // derived class of the current instantiation of the hierarchy. This
    // is a slight variant of the CRTP where the derived class passes 
    // itself as a template argument to the base class(es). Here, it seemed
    // more convenient to pass up just the traversal tag of the most 
    // derived class.
    //
    template<
      class WrappedIterator,
      class Value,
      class Traversal,
      class Reference,
      class Difference,
      class UsedAsBaseForTraversal = Traversal
    >
    class any_iterator_wrapper;

    ///////////////////////////////////////////////////////////////////////
    // 
    template<
      class WrappedIterator,
      class Value,
      class Reference,
      class Difference,
      class UsedAsBaseForTraversal
    >
    class any_iterator_wrapper<
      WrappedIterator,
      Value,
      boost::incrementable_traversal_tag,
      Reference,
      Difference,
      UsedAsBaseForTraversal
    > : public any_iterator_abstract_base<
          Value,
          UsedAsBaseForTraversal,
          Reference,
          Difference
        >
    {

    protected:
      typedef any_iterator_abstract_base<Value, UsedAsBaseForTraversal, Reference, Difference> abstract_base_type;

    private:
      typedef typename abstract_base_type::clone_result_type clone_result_type;
      typedef typename abstract_base_type::const_clone_with_const_value_type_result_type const_clone_with_const_value_type_result_type;
      typedef typename abstract_base_type::const_clone_with_non_const_value_type_result_type const_clone_with_non_const_value_type_result_type;

      typedef any_iterator_wrapper<
        WrappedIterator,
        Value,
        UsedAsBaseForTraversal,
        Reference,
        Difference
      > clone_type;

      typedef any_iterator_wrapper<
        WrappedIterator,
        typename boost::add_const<Value>::type,
        UsedAsBaseForTraversal,
        typename make_iterator_reference_const<Reference>::type,
        Difference
      > const_clone_type_with_const_value_type;

      typedef any_iterator_wrapper<
        WrappedIterator,
        typename boost::remove_const<Value>::type,
        UsedAsBaseForTraversal,
        typename make_iterator_reference_const<Reference>::type,
        Difference
      > const_clone_type_with_non_const_value_type;

    public:

      any_iterator_wrapper()
      {}

      any_iterator_wrapper(WrappedIterator const& wrapped_iterator) :
        m_wrapped_iterator(wrapped_iterator)
      {}

      static abstract_base_type* create(WrappedIterator const& wrapped_iterator)
      {
        return new clone_type(wrapped_iterator);
      }

      // Plain clone function for copy construction and assignment.
      virtual clone_result_type * clone() const
      {
        return new clone_type(m_wrapped_iterator);
      }
  
      // Clone functions for conversion to a const iterator
      virtual const_clone_with_const_value_type_result_type* make_const_clone_with_const_value_type() const
      {
        return new const_clone_type_with_const_value_type(m_wrapped_iterator);
      }
      //
      virtual const_clone_with_non_const_value_type_result_type* make_const_clone_with_non_const_value_type() const
      {
        return new const_clone_type_with_non_const_value_type(m_wrapped_iterator);
      }
      
      virtual Reference dereference() const
      {
        // This const cast is needed for output iterators. Is this perhaps an oversight
        // in iterator_facade?
        return *const_cast<any_iterator_wrapper*>(this)->m_wrapped_iterator;
      }

      virtual void increment()
      {
        ++m_wrapped_iterator;
      }

    protected:

      WrappedIterator& get_wrapped_iterator()
      {
        return m_wrapped_iterator;
      }

      WrappedIterator const & get_wrapped_iterator() const
      {
        return m_wrapped_iterator;
      }

    private:

      WrappedIterator m_wrapped_iterator;

    };

    ///////////////////////////////////////////////////////////////////////
    // 
    template<
      class WrappedIterator,
      class Value,
      class Reference,
      class Difference,
      class UsedAsBaseForTraversal
    >
    class any_iterator_wrapper<
      WrappedIterator,
      Value,
      boost::single_pass_traversal_tag,
      Reference,
      Difference,
      UsedAsBaseForTraversal
    > : public any_iterator_wrapper<
          WrappedIterator,
          Value,
          boost::incrementable_traversal_tag,
          Reference,
          Difference,
          UsedAsBaseForTraversal
        >
    {

    public:
      
      typedef
      any_iterator_wrapper<
        WrappedIterator,
        Value,
        boost::incrementable_traversal_tag,
        Reference,
        Difference,
        UsedAsBaseForTraversal
      > super_type;

      any_iterator_wrapper()
      {}

      any_iterator_wrapper(WrappedIterator const& wrapped_iterator) : super_type(wrapped_iterator)
      {}

      // gcc 3.4.2 insists on qualification of abstract_base_type.
      virtual bool equal(typename any_iterator_wrapper::abstract_base_type const & rhs) const
      {
        return this->get_wrapped_iterator() == boost::polymorphic_downcast<any_iterator_wrapper const *>(&rhs)->get_wrapped_iterator();
      }

      any_iterator_abstract_base<
        Value,
        boost::incrementable_traversal_tag,
        Reference,
        Difference
      >* make_incrementable_version() const
      {
        return new any_iterator_wrapper<
          WrappedIterator,
          Value,
          boost::incrementable_traversal_tag,
          Reference,
          Difference
        >(this->get_wrapped_iterator());
      }
    };

    ///////////////////////////////////////////////////////////////////////
    // 
    template<
      class WrappedIterator,
      class Value,
      class Reference,
      class Difference,
      class UsedAsBaseForTraversal
    >
    class any_iterator_wrapper<
      WrappedIterator,
      Value,
      boost::forward_traversal_tag,
      Reference,
      Difference,
      UsedAsBaseForTraversal
    > : public any_iterator_wrapper<
          WrappedIterator,
          Value,
          boost::single_pass_traversal_tag,
          Reference,
          Difference,
          UsedAsBaseForTraversal
        >
    {

    public:
      
      typedef
      any_iterator_wrapper<
        WrappedIterator,
        Value,
        boost::single_pass_traversal_tag,
        Reference,
        Difference,
        UsedAsBaseForTraversal
      > super_type;

      any_iterator_wrapper()
      {}

      any_iterator_wrapper(WrappedIterator const& wrapped_iterator) : super_type(wrapped_iterator)
      {}

      any_iterator_abstract_base<
        Value,
        boost::single_pass_traversal_tag,
        Reference,
        Difference
      >* make_single_pass_version() const
      {
        return new any_iterator_wrapper<
          WrappedIterator,
          Value,
          boost::single_pass_traversal_tag,
          Reference,
          Difference
        >(this->get_wrapped_iterator());
      }
    };

    ///////////////////////////////////////////////////////////////////////
    // 
    template<
      class WrappedIterator,
      class Value,
      class Reference,
      class Difference,
      class UsedAsBaseForTraversal
    >
    class any_iterator_wrapper<
      WrappedIterator,
      Value,
      boost::bidirectional_traversal_tag,
      Reference,
      Difference,
      UsedAsBaseForTraversal
    > : public any_iterator_wrapper<
          WrappedIterator,
          Value,
          boost::forward_traversal_tag,
          Reference,
          Difference,
          UsedAsBaseForTraversal
        >
    {

    public:
      
      typedef
      any_iterator_wrapper<
        WrappedIterator,
        Value,
        boost::forward_traversal_tag,
        Reference,
        Difference,
        UsedAsBaseForTraversal
      > super_type;

      any_iterator_wrapper()
      {}

      any_iterator_wrapper(WrappedIterator const& wrapped_iterator) : super_type(wrapped_iterator)
      {}

      virtual void decrement()
      {
        --(this->get_wrapped_iterator());
      }

      any_iterator_abstract_base<
        Value,
        boost::forward_traversal_tag,
        Reference,
        Difference
      >* make_forward_version() const
      {
        return new any_iterator_wrapper<
          WrappedIterator,
          Value,
          boost::forward_traversal_tag,
          Reference,
          Difference
        >(this->get_wrapped_iterator());
      }
    };

    ///////////////////////////////////////////////////////////////////////
    // 
    template<
      class WrappedIterator,
      class Value,
      class Reference,
      class Difference,
      class UsedAsBaseForTraversal
    >
    class any_iterator_wrapper<
      WrappedIterator,
      Value,
      boost::random_access_traversal_tag,
      Reference,
      Difference,
      UsedAsBaseForTraversal
    > : public any_iterator_wrapper<
          WrappedIterator,
          Value,
          boost::bidirectional_traversal_tag,
          Reference,
          Difference,
          UsedAsBaseForTraversal
        >
    {

    public:
      
      typedef
      any_iterator_wrapper<
        WrappedIterator,
        Value,
        boost::bidirectional_traversal_tag,
        Reference,
        Difference,
        UsedAsBaseForTraversal
      > super_type;

      any_iterator_wrapper()
      {}

      any_iterator_wrapper(WrappedIterator const& wrapped_iterator) : super_type(wrapped_iterator)
      {}

      virtual void advance(Difference n)
      {
        this->get_wrapped_iterator() += n;
      }

      // gcc 3.4.2 insists on qualification of abstract_base_type.
      virtual Difference distance_to(typename any_iterator_wrapper::abstract_base_type const & other) const
      {
        return boost::polymorphic_downcast<any_iterator_wrapper const *>(&other)->get_wrapped_iterator() - this->get_wrapped_iterator();
      }

      any_iterator_abstract_base<
        Value,
        boost::bidirectional_traversal_tag,
        Reference,
        Difference
      >* make_bidirectional_version() const
      {
        return new any_iterator_wrapper<
          WrappedIterator,
          Value,
          boost::bidirectional_traversal_tag,
          Reference,
          Difference
        >(this->get_wrapped_iterator());
      }
    };

  } // end namespace detail

} // end namespace IteratorTypeErasure

#endif // ANY_ITERATOR_WRAPPER_01102007TMB_HPP
