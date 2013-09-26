//  (C) Copyright Thomas Becker 2005. Permission to copy, use, modify, sell and
//  distribute this software is granted provided this copyright notice appears
//  in all copies. This software is provided "as is" without express or implied
//  warranty, and with no claim as to its suitability for any purpose.

// File Name
// =========
//
// metafunctions.h

// Description
// ===========
//
// Metafunctions for any_iterator

#ifndef ANY_ITERATOR_METAFUNCTIONS_01102007TMB_HPP
#define ANY_ITERATOR_METAFUNCTIONS_01102007TMB_HPP

// Revision History
// ================
//
// 27 Dec 2006 (Thomas Becker) Created

// Includes
// ========
#include <boost/iterator/iterator_categories.hpp>
#include <boost/iterator/iterator_traits.hpp>
#include <boost/type_traits/remove_const.hpp>
#include <boost/type_traits/remove_reference.hpp>
#include <boost/type_traits/is_same.hpp>
#include <boost/type_traits/is_base_of.hpp>
#include <boost/type_traits/is_convertible.hpp>
#include <boost/type_traits/is_reference.hpp>
#include <boost/type_traits/is_pointer.hpp>
#include <boost/mpl/bool.hpp>
#include <boost/mpl/and.hpp>
#include <boost/mpl/or.hpp>
#include <boost/mpl/not.hpp>

namespace IteratorTypeErasure
{

  template<
    class Value,
    class Traversal,
    class Reference,
    class Difference
    >
  class any_iterator;

  namespace detail
  {

    ///////////////////////////////////////////////////////////////////////
    // 
    template<typename T>
    struct remove_reference_and_const
    {
      typedef typename boost::remove_const<
        typename boost::remove_reference<
          typename boost::remove_const<
            T
          >::type
        >::type
      >::type type;
    };
    
    ///////////////////////////////////////////////////////////////////////
    // 
    template<typename IteratorReference>
    struct make_iterator_reference_const
    {
      typedef typename boost::mpl::if_< 
        typename boost::is_reference<IteratorReference>::type,
        typename boost::remove_const<
          typename boost::remove_reference<
            typename boost::remove_const<
              IteratorReference
              >::type
            >::type
          >::type const &,
        typename boost::remove_const<
          typename boost::remove_reference<
            typename boost::remove_const<
              IteratorReference
              >::type
            >::type
          >::type const
        >::type type;
    };

    ///////////////////////////////////////////////////////////////////////
    // 
    template<
      class WrappedIterator,
      class AnyIterator
    >
    struct value_types_erasure_compatible
    {
      // Really, we just want WrappedIterator's value_type to convert to
      // AnyIterator's value_type. But many real world output iterators
      // define their value_type as void. Therefore, we simply ignore
      // the value type for output iterators. That's fine because for
      // output iterators, the relevant type erasure information is all
      // in the reference type.

      typedef typename boost::mpl::or_<
        boost::is_same<
          typename boost::iterator_category<WrappedIterator>::type,
          std::output_iterator_tag
        >,

        // Really, we just want WrappedIterator's value_type to convert to
        // AnyIterator's value_type. But we need to work around a flaw in
        // boost::is_convertible. boost::is_convertible<X, Y>::value is
        // false whenever X and Y are abstract base classes (even when X
        // and Y are the same). This will be fixed in C++ concepts.
        boost::mpl::or_<
          boost::is_same<
            typename boost::iterator_value<WrappedIterator>::type,
            typename boost::iterator_value<AnyIterator>::type
          >,
          boost::is_base_of<
            typename boost::iterator_value<AnyIterator>::type,
            typename boost::iterator_value<WrappedIterator>::type
          >,
          boost::is_convertible<
            typename boost::iterator_value<WrappedIterator>::type,
            typename boost::iterator_value<AnyIterator>::type
          >
        >
      > type;
    };

    ///////////////////////////////////////////////////////////////////////
    // 
    template<
      class WrappedIterator,
      class AnyIterator
    >
    struct reference_types_erasure_compatible_1
    {
      typedef typename boost::is_convertible<
        typename boost::iterator_reference<WrappedIterator>::type,
        typename boost::iterator_reference<AnyIterator>::type
      >::type type;
    };

    ///////////////////////////////////////////////////////////////////////
    // 
    template<
      class WrappedIterator,
      class AnyIterator
    >
    struct reference_types_erasure_compatible_2
    {
      typedef typename boost::mpl::if_<
        boost::is_reference<
          typename boost::iterator_reference<AnyIterator>::type
        >,
        boost::is_reference<
          typename boost::iterator_reference<WrappedIterator>::type
        >,
        boost::mpl::bool_<true>
      >::type type;
    };

    ///////////////////////////////////////////////////////////////////////
    // 
    template<
      class WrappedIterator,
      class AnyIterator
    >
    struct reference_types_erasure_compatible_3
    {
      
      typedef typename boost::mpl::if_<
        boost::mpl::and_<
          boost::is_reference<
            typename boost::iterator_reference<AnyIterator>::type
          >,
          boost::is_reference<
            typename boost::iterator_reference<WrappedIterator>::type
          >
        >,
        boost::mpl::or_<
          boost::is_same<
            typename remove_reference_and_const<
              typename boost::iterator_reference<AnyIterator>::type
            >::type,
            typename remove_reference_and_const<
              typename boost::iterator_reference<WrappedIterator>::type
            >::type
          >,
          boost::is_base_of<
            typename remove_reference_and_const<
              typename boost::iterator_reference<AnyIterator>::type
            >::type,
            typename remove_reference_and_const<
              typename boost::iterator_reference<WrappedIterator>::type
            >::type
          >
        >,
        boost::mpl::bool_<true>
      >::type type;
    };

    ///////////////////////////////////////////////////////////////////////
    // 
    template<
      class WrappedIterator,
      class AnyIterator
    >
    struct reference_types_erasure_compatible
    {
      // Output iterators are weird. Many real world output iterators
      // define their reference type as void. In the world of boost
      // iterators, that's terribly wrong, because in that world, an
      // iterator's reference type is always the result type of
      // operator* (and that makes very good sense, too). Therefore,
      // when WrappedIterator is an output iterator, we use
      // WrappedIterator& for WrappedIterator's reference type,
      // because that's the real, true reference type. Moreover,
      // we just require that WrappedIterator& convert to AnyIterator's
      // reference type. The other subtleties are not relevant.

      typedef typename boost::mpl::if_<
        boost::is_same<
          typename boost::iterator_category<WrappedIterator>::type,
          std::output_iterator_tag
        >,
        boost::is_convertible<
          WrappedIterator&,
          typename boost::iterator_reference<AnyIterator>::type
        >,
        boost::mpl::and_<

          // WrappedIterator's reference type must convert to AnyIterator's reference type.
          typename reference_types_erasure_compatible_1<WrappedIterator, AnyIterator>::type,

          // If AnyIterator's reference type is a reference, then the
          // same must be true for WrappedIterator's reference type.
          typename reference_types_erasure_compatible_2<WrappedIterator, AnyIterator>::type,

          // If AnyIterator's reference type and WrappedIterator's
          // reference type are both references, then one of the
          // following must hold:
          //
          // 1) AnyIterator's reference type and WrappedIterator's
          //    reference type are the same.
          //
          // 2) AnyIterator's reference type is a base class of WrappedIterator's
          //    reference type.
          //
          typename reference_types_erasure_compatible_3<WrappedIterator, AnyIterator>::type
        >
      >::type type;
    };

    ///////////////////////////////////////////////////////////////////////
    // 
    template<
      class WrappedIterator,
      class AnyIterator
    >
    struct difference_types_erasure_compatible
    {

      // Difference type matters only for random access iterators.
      
      typedef typename boost::mpl::or_<
        boost::mpl::not_<
          boost::is_same<
            // Do not use boost::iterator_traversal<AnyIterator>::type here,
            // as it does not equal the traversal tag.
            typename AnyIterator::Traversal,
            boost::random_access_traversal_tag
          >
        >,
        boost::mpl::and_<
          boost::is_convertible<
            typename boost::iterator_difference<WrappedIterator>::type,
            typename boost::iterator_difference<AnyIterator>::type
          >,
          boost::is_convertible<
            typename boost::iterator_difference<AnyIterator>::type,
            typename boost::iterator_difference<WrappedIterator>::type
          >
        >
      > type;
    };

    ///////////////////////////////////////////////////////////////////////
    // 
    template<
      class WrappedIterator,
      class AnyIterator
    >
    struct traversal_types_erasure_compatible
    {
      typedef typename boost::mpl::or_<
        boost::is_same<
          // Do not use boost::iterator_traversal<AnyIterator>::type here,
          // as it does not equal the traversal tag.
          typename AnyIterator::Traversal,
          typename boost::iterator_traversal<WrappedIterator>::type
        >,
        boost::is_base_of<
          // Do not use boost::iterator_traversal<AnyIterator>::type here,
          // as it does not equal the traversal tag.
          typename AnyIterator::Traversal,
          typename boost::iterator_traversal<WrappedIterator>::type
        >
      > type;
    };

    ///////////////////////////////////////////////////////////////////////
    // 
    template<
      class WrappedIterator,
      class AnyIterator
    >
    struct is_iterator_type_erasure_compatible :
      public boost::mpl::bool_<
        boost::mpl::and_<
          value_types_erasure_compatible<WrappedIterator, AnyIterator>,
          reference_types_erasure_compatible<WrappedIterator, AnyIterator>,
          difference_types_erasure_compatible<WrappedIterator, AnyIterator>,
          traversal_types_erasure_compatible<WrappedIterator, AnyIterator>
        >::value
      >
    {
    };

    ///////////////////////////////////////////////////////////////////////
    // 
    template<class SomeIterator>
    struct is_any_iterator : public boost::mpl::bool_<false>
    {
    };
    //
    template<
      class Value,
      class Traversal,
      class Reference,
      class Difference
      >
    struct is_any_iterator<
      any_iterator<
        Value,
        Traversal,
        Reference,
        Difference
      >
    > : public boost::mpl::bool_<true>
    {
    };
    
  } // end namespace detail

} // end namespace IteratorTypeErasure

#endif // ANY_ITERATOR_METAFUNCTIONS_01102007TMB_HPP
